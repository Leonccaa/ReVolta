# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary ownership and loader integrity contracts for sharded codebooks."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3_dflash2 as dflash2

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _lookup(table, ids, tp):
    lookup = getattr(dflash2, "_lookup_sharded_rows", None)
    if lookup is None:
        return table[ids]
    rows = -(-table.shape[0] // tp)
    total = None
    for rank in range(tp):
        shard = torch.zeros(rows, table.shape[1], dtype=table.dtype)
        part = table[rank * rows : (rank + 1) * rows]
        shard[: len(part)] = part
        result = lookup(shard, ids, rank * rows, reduce=lambda x: x)
        total = result if total is None else total + result
    return total


@pytest.mark.parametrize("vocab,tp", [(3, 4), (17, 4), (20, 4), (248320, 4)])
def test_all_valid_ids_have_exactly_one_owner_and_match_replicated(vocab, tp):
    rows = -(-vocab // tp)
    ids = torch.arange(vocab)
    owners = sum(((ids >= r * rows) & (ids < (r + 1) * rows)).int() for r in range(tp))
    assert (owners == 1).all()
    table = (torch.arange(vocab * 3).reshape(vocab, 3) % 997).half()
    assert torch.equal(
        _lookup(table, ids, tp).view(torch.int16), table.view(torch.int16)
    )


def test_all_reduce_of_zeros_preserves_codebook_bits_including_negative_zero():
    table = torch.tensor([[1.0, -0.0], [-0.0, -2.0], [3.0, -0.0]]).half()
    actual = _lookup(table, torch.arange(3), 4)
    assert torch.equal(actual.view(torch.int16), table.view(torch.int16)), (
        "Floating-point addition of positive zeros changes negative-zero bits"
    )


@pytest.mark.parametrize("shape", [(16, 3), (18, 3), (17, 1)])
def test_loader_rejects_mismatched_full_checkpoint_shape(shape):
    loaded = torch.ones(shape)
    loader = getattr(dflash2.CandidateSelector, "_load_rows", None)
    if loader is None:
        param = torch.nn.Parameter(torch.empty(17, 3), requires_grad=False)
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            default_weight_loader(param, loaded)
    else:
        owner = SimpleNamespace(vocab_size=17, vocab_start=15, rows_per_rank=5)
        param = torch.nn.Parameter(torch.empty(5, 3), requires_grad=False)
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            loader(owner, param, loaded)


def test_valid_loader_preserves_last_real_row_and_zero_pads_tail():
    loader = getattr(dflash2.CandidateSelector, "_load_rows", None)
    if loader is None:
        pytest.skip("Row-slicing loader is absent on the baseline")
    assert loader is not None
    full = torch.arange(17 * 3).reshape(17, 3).float()
    owner = SimpleNamespace(vocab_size=17, vocab_start=15, rows_per_rank=5)
    param = torch.nn.Parameter(torch.empty(5, 3), requires_grad=False)
    loader(owner, param, full)
    assert torch.equal(param[:2], full[15:])
    assert (param[2:] == 0).all()
