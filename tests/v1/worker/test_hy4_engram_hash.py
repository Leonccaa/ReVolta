# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashEngramLayer

from vllm.models.hy_v4_flash.common.ngram_hash import (
    hash_ngrams_torch,
    hash_ngrams_triton,
)
from vllm.models.hy_v4_flash.nvidia.engram import FlashNgramHashState

pytestmark = pytest.mark.skip_global_cleanup


def make_case():
    torch.manual_seed(123)
    layers = (2, 12, 22, 32)
    sizes = [[11, 13], [17, 19]]
    config = SimpleNamespace(
        hidden_size=8,
        hc_mult=2,
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=8,
        engram_fp8_embedding=False,
        engram_pad_id=3,
        engram_seed=42,
        engram_layer_head_vocab_sizes={str(layer): sizes for layer in layers},
        engram_lookup_table_size=32,
    )
    references = [HYV4FlashEngramLayer(config, layer) for layer in layers]
    maps = []
    for reference in references:
        mapping = torch.randint(0, 25, (32,), dtype=torch.int32)
        reference.lookup_table.copy_(mapping)
        reference.hash_multipliers.copy_(
            reference.draw_hash_multipliers(int(mapping.max()) + 1)
        )
        maps.append(mapping)
    maps_tensor = torch.stack(maps)
    multipliers = torch.stack([ref.hash_multipliers for ref in references])
    primes = torch.tensor([[11, 13, 17, 19]] * 4, dtype=torch.int64)
    offsets = torch.tensor([[0, 11, 24, 41]] * 4, dtype=torch.int64)
    return references, maps_tensor, torch.full((4,), 32), multipliers, primes, offsets


@pytest.mark.parametrize("seed", [10, 25, 40, 80])
def test_random_hash_against_transformers(seed):
    refs, maps, sizes, multipliers, primes, offsets = make_case()
    tokens = torch.randint(
        0, 45, (11,), generator=torch.Generator().manual_seed(seed), dtype=torch.int32
    )
    tokens[0], tokens[3], tokens[7] = 4, 3, 35
    starts = torch.tensor([0, 7, 11], dtype=torch.int32)
    history = torch.tensor([[3, 3], [29, 34]], dtype=torch.int32)
    result = hash_ngrams_torch(
        tokens, starts, history, maps, sizes, multipliers, primes, offsets, 3, 11
    )
    for layer, ref in enumerate(refs):
        expected_rows = []
        for begin, end, previous in ((0, 7, [3, 3]), (7, 11, [29, 34])):
            seq = previous + tokens[begin:end].tolist()
            grams = torch.tensor(
                [[seq[i], seq[i - 1], seq[i - 2]] for i in range(2, len(seq))]
            )
            expected_rows.append(ref.hash_ngrams(grams) + offsets[layer])
        expected = torch.cat(expected_rows).int()
        expected[tokens == 3] = -1
        torch.testing.assert_close(result[layer], expected, rtol=0, atol=0)


def test_chunks_match_full_sequence():
    _, maps, sizes, multipliers, primes, offsets = make_case()
    tokens = torch.tensor([8, 9, 35, 3, 7, 25, 13, 31, 5], dtype=torch.int32)

    def compute(chunk, history):
        return hash_ngrams_torch(
            chunk,
            torch.tensor([0, len(chunk)], dtype=torch.int32),
            history,
            maps,
            sizes,
            multipliers,
            primes,
            offsets,
            3,
            len(chunk),
        )

    full = compute(tokens, torch.tensor([[3, 3]], dtype=torch.int32))
    pieces = []
    for start, end in ((0, 1), (1, 4), (4, 5), (5, 9)):
        history = ([3, 3] + tokens[:start].tolist())[-2:]
        pieces.append(
            compute(tokens[start:end], torch.tensor([history], dtype=torch.int32))
        )
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=0, atol=0)


def test_negative_and_padded_input():
    _, maps, sizes, multipliers, primes, offsets = make_case()
    starts = torch.tensor([0, 2, 2], dtype=torch.int32)
    history = torch.tensor([[3, 3], [3, 3]], dtype=torch.int32)
    args = (starts, history, maps, sizes, multipliers, primes, offsets)
    # The captured token count includes padding. The device offsets give the
    # real end even when the padded input is a non-pad ID from a previous step.
    actual = hash_ngrams_torch(torch.tensor([-1, 4, 7]), *args, 3, 3)
    expected = hash_ngrams_torch(torch.tensor([3, 4, 3]), *args, 3, 2)
    torch.testing.assert_close(actual[:, 1:], expected[:, 1:], rtol=0, atol=0)
    assert (actual[:, 2] == -1).all()


def test_hash_state_loads_each_layer_and_writes_output(monkeypatch):
    refs, maps, sizes, multipliers, primes, offsets = make_case()
    layers = (2, 12, 22, 32)
    config = SimpleNamespace(
        engram_layer_ids=layers,
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=8,
        engram_pad_id=3,
        engram_seed=42,
        vocab_size=32,
        engram_layer_head_vocab_sizes={
            str(layer): [[11, 13], [17, 19]] for layer in layers
        },
    )
    state = FlashNgramHashState(config)
    for index, layer in enumerate(layers):
        state.load_lookup(maps[index], layer)
    torch.testing.assert_close(state.multipliers, multipliers, rtol=0, atol=0)
    tokens = torch.tensor([4, 31, 35, 3, 8], dtype=torch.int32)
    history = torch.tensor([[3, 3]], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 5]), num_actual_tokens=5
    )
    monkeypatch.setattr(
        "vllm.models.hy_v4_flash.nvidia.engram.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"attn": metadata}),
    )
    output = torch.empty((4, 5, 4), dtype=torch.int32)
    state.prepare_hashes_with_output(tokens, history, "attn", output)
    expected = hash_ngrams_torch(
        tokens,
        metadata.query_start_loc,
        history,
        maps,
        sizes,
        multipliers,
        primes,
        offsets,
        3,
        5,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(refs) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_matches_cpu():
    _, maps, sizes, multipliers, primes, offsets = make_case()
    tokens = torch.tensor([4, 31, 35, 3, 8, 10, 1, 16, 3, 30, 38, 8], dtype=torch.int32)
    starts = torch.tensor([0, 7, 11, 11], dtype=torch.int32)
    history = torch.tensor([[3, 3], [29, 34], [3, 3]], dtype=torch.int32)
    args = (tokens, starts, history, maps, sizes, multipliers, primes, offsets)
    expected = hash_ngrams_torch(*args, 3, 11)
    result = hash_ngrams_triton(*(arg.cuda() for arg in args), 3, 12)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
