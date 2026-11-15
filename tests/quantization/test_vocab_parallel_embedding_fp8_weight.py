# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Float8 checkpoint weights must have a float8 destination table."""

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from vllm.model_executor.layers import vocab_parallel_embedding as vocab

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture(params=[vocab.VocabParallelEmbedding, vocab.ParallelLMHead])
def layer(request, monkeypatch):
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 1)
    prefix = (
        "model.embed_tokens"
        if request.param is vocab.VocabParallelEmbedding
        else "lm_head"
    )
    return request.param(64, 16, params_dtype=torch.bfloat16, prefix=prefix)


@pytest.mark.parametrize("dtype", vocab._FLOAT8_DTYPES)
def test_float8_weight_is_refused(layer, dtype):
    weight = torch.ones(64, 16).to(dtype)
    with pytest.raises(ValueError, match=layer.prefix):
        layer.weight_loader(layer.weight, weight)


def test_unquantized_weight_still_loads(layer):
    weight = torch.arange(1024).reshape(64, 16).to(torch.bfloat16)
    layer.weight_loader(layer.weight, weight)
    torch.testing.assert_close(layer.weight, weight)


def test_float8_table_and_uninitialized_parameter_still_load(layer):
    weight = torch.ones(64, 16).to(torch.float8_e4m3fn)
    for param in (
        torch.nn.Parameter(torch.empty_like(weight), requires_grad=False),
        UninitializedParameter(requires_grad=False),
    ):
        param.output_dim = 0
        layer.weight_loader(param, weight)
        assert param.dtype == weight.dtype
        torch.testing.assert_close(param.float(), weight.float())
