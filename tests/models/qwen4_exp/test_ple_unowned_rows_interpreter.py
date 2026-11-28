# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE gathers write unowned rows as zeros without reading the table.

Rows owned by another TP rank, or by the other half of a device/host split,
used to be redirected to row 0 and read; on a pinned host table those
same-address reads serialize over PCIe. These tests run the gather in the
Triton CPU interpreter.

Run with: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= pytest <this file>
"""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "Triton CPU interpreter test (set TRITON_INTERPRET=1)",
        allow_module_level=True,
    )

import torch

from vllm.models.qwen4_exp.nvidia import ple_layer

DIM = 32
SCALE = 0.5


def _table(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    table = torch.randint(0, 0x7E, (rows, DIM), generator=gen, dtype=torch.uint8)
    # Negative zero must survive the gather and the merges bit-for-bit.
    table[:, 0] = 0x80
    return table


def _reference(table: torch.Tensor, ids: list[int]) -> torch.Tensor:
    rows = [
        table[i].view(torch.float8_e4m3fn).float() * SCALE
        if i >= 0
        else torch.zeros(DIM)
        for i in ids
    ]
    return torch.stack(rows).bfloat16()


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_gather_writes_zeros_for_negative_ids():
    table = _table(4, seed=0)
    ids = [2, -1, 0, -1, 3, 2]
    out = torch.full((len(ids), DIM), float("nan"), dtype=torch.bfloat16)
    ple_layer._gather_ple_fp8_from_pinned_kernel[(len(ids),)](
        table,
        torch.tensor(ids, dtype=torch.int64),
        torch.tensor([SCALE], dtype=torch.float32),
        out,
        embedding_dim=DIM,
        BLOCK_D=DIM,
    )
    _assert_bitwise(out, _reference(table, ids))


@pytest.mark.parametrize(("device_rows", "host_rows"), [(0, 6), (6, 0), (2, 4)])
def test_tp_forward_zeroes_other_ranks_rows_across_the_split(
    monkeypatch, device_rows, host_rows
):
    local = _table(device_rows + host_rows, seed=1)
    device_table = local[:device_rows].contiguous()
    host_table = local[device_rows:].contiguous()
    layer = SimpleNamespace(
        tp_size=2,
        # Rank 1 of 2 owns global ids [6, 12).
        shard_indices=SimpleNamespace(
            org_vocab_start_index=6,
            org_vocab_end_index=12,
            num_org_vocab_padding=0,
            added_vocab_start_index=12,
            added_vocab_end_index=12,
        ),
        _accelerator_weight_ptrs={0: host_table.data_ptr()},
        _device_table_ptr=device_table.data_ptr(),
        _device_rows=device_rows,
        _host_rows=host_rows,
        _disk_rows=0,
        embedding_dim=DIM,
        _output_dtype=torch.bfloat16,
        weight_scale=torch.tensor([SCALE], dtype=torch.float32),
        layer_name="ple",
    )
    layer.embedding_lookup = lambda ids, remote_rows=None: (
        ple_layer.Qwen4ExpPinnedHostEmbedding.embedding_lookup(layer, ids, remote_rows)
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        ple_layer,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    monkeypatch.setattr(ple_layer, "tensor_model_parallel_all_reduce", lambda t: t)

    gather = torch.ops.vllm.qwen4_exp_ple_pinned_gather
    gathered_ids = {}

    def record_gather(ids, output, scale, name, use_host, dim):
        gathered_ids[use_host] = ids.clone()
        gather(ids, output, scale, name, use_host, dim)

    monkeypatch.setattr(torch.ops.vllm, "qwen4_exp_ple_pinned_gather", record_gather)

    global_ids = [[6, 0, 11], [3, 7, 9], [8, 8, 5]]
    out = ple_layer.Qwen4ExpPinnedHostEmbedding.forward(layer, torch.tensor(global_ids))

    local_ids = [i - 6 if 6 <= i < 12 else -1 for row in global_ids for i in row]
    _assert_bitwise(out.reshape(-1, DIM), _reference(local, local_ids))
    if device_rows and host_rows:
        # The final merge hides reads of rows that belong to the other table.
        on_host = torch.tensor(local_ids) >= device_rows
        assert (gathered_ids[False][on_host] < 0).all()
        assert (gathered_ids[True][~on_host] < 0).all()
