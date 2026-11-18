# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
    HYV4FlashEngramLayer,
    _fp8_round_trip,
)

import vllm.models.hy_v4_flash.nvidia.engram as engram

pytestmark = pytest.mark.skip_global_cleanup


def make_case(tmp_path):
    torch.manual_seed(77)
    config = SimpleNamespace(
        hidden_size=8,
        hc_mult=2,
        engram_layer_ids=[2],
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=8,
        engram_fp8_embedding=True,
        engram_pad_id=3,
        engram_seed=42,
        engram_layer_head_vocab_sizes={"2": [[11, 13], [17, 19]]},
        engram_lookup_table_size=32,
        vocab_size=32,
    )
    reference = HYV4FlashEngramLayer(config, 2)
    source = torch.randn(60, 4, dtype=torch.bfloat16)
    scale = source.abs().amax(-1).clamp(min=1e-12) / 448
    codes = (source / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    with torch.no_grad():
        reference.multi_head_embedding.embedding.weight.copy_(source.float())
        reference.unified_pre_gamma.uniform_(0.5, 1.5)
        for item in reference.unified_per_hc_scales:
            item.uniform_(-0.2, 0.2)
        for module in [
            *reference.lookup_per_head_norms,
            *reference.norm_key,
            *reference.norm_query,
        ]:
            module.weight.uniform_(0.5, 1.5)
    path = tmp_path / "table.safetensors"
    save_file({"table.weight": codes, "table.weight_scale_inv": scale}, path)
    layout = engram.FlashEngramLayout(config)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        device_config=SimpleNamespace(device_type="cpu"),
        load_config=SimpleNamespace(load_format="safetensors"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    tokens = torch.tensor([4, 5, 31, 3, 8, 6])
    shifted = [3, 3] + tokens.tolist()
    ngrams = torch.tensor(
        [[shifted[i], shifted[i - 1], shifted[i - 2]] for i in range(2, len(shifted))]
    )
    hashes = reference.hash_ngrams(ngrams) + reference.multi_head_embedding.offsets
    hashes[tokens == config.engram_pad_id] = -1
    hidden = torch.randn(len(tokens), config.hc_mult, config.hidden_size)
    return (
        reference,
        source,
        codes,
        scale,
        path,
        layout,
        vllm_config,
        tokens,
        ngrams,
        hashes,
        hidden,
    )


def make_layer(reference, path, layout, vllm_config, tp, rank, monkeypatch):
    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: rank)
    layer = engram.FlashEngram(vllm_config, layout, 2, f"engram_{rank}")
    layer.multi_head_embedding.load_from_safetensors(str(path), "table.weight")
    with torch.no_grad():
        for i, norm in enumerate(layer.lookup_per_head_norms):
            norm.weight.copy_(
                reference.lookup_per_head_norms[
                    layer.multi_head_embedding.head_start + i
                ].weight
            )
        for dst, src in zip(
            [layer.value_proj, *layer.key_projs],
            [reference.value_proj, *reference.key_projs],
        ):
            dst.weight.weight_loader(dst.weight, src.weight)
            dst.bias.weight_loader(dst.bias, src.bias)
        for dst, src in zip(layer.norm_key, reference.norm_key):
            dst.weight.copy_(src.weight)
        for dst, src in zip(layer.norm_query, reference.norm_query):
            dst.weight.copy_(src.weight)
        layer.unified_pre_gamma.copy_(reference.unified_pre_gamma)
        for dst, src in zip(
            layer.unified_per_hc_scales, reference.unified_per_hc_scales
        ):
            dst.copy_(src)
    return layer


def test_fp8_row_decode_against_reference(tmp_path):
    _, source, codes, scale, *_ = make_case(tmp_path)
    actual = (codes.float() * scale.float()[:, None]).to(torch.bfloat16).float()
    expected = _fp8_round_trip(source).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_whole_layer_against_transformers(tmp_path, monkeypatch):
    (
        reference,
        source,
        codes,
        scale,
        path,
        layout,
        cfg,
        tokens,
        ngrams,
        hashes,
        hidden,
    ) = make_case(tmp_path)
    layer = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch)
    actual = layer(hidden, hashes) - hidden
    mask = (tokens != 3)[None]
    expected = reference(hidden[None], ngrams[None], mask)[0]
    error = actual - expected
    # The reference quantizes FP32 inputs here, unlike the offline BF16 recipe.
    assert error.norm() / expected.norm() < 0.01
    assert error.abs().max() < 0.02
    reference.fp8_embedding = False
    with torch.no_grad():
        reference.multi_head_embedding.embedding.weight.copy_(
            _fp8_round_trip(source).float()
        )
    expected_bf16 = reference(hidden[None], ngrams[None], mask)[0]
    torch.testing.assert_close(actual, expected_bf16, rtol=1e-5, atol=3e-7)
    torch.testing.assert_close(
        layer(hidden, hashes)[tokens == 3], hidden[tokens == 3], rtol=0, atol=0
    )


def test_fp16_whole_layer_against_transformers(tmp_path, monkeypatch):
    reference, _, _, _, path, layout, cfg, tokens, ngrams, hashes, hidden = make_case(
        tmp_path
    )
    layer = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch).half()
    reference.half()
    hidden = hidden.half()
    expected = hidden + reference(hidden[None], ngrams[None], (tokens != 3)[None])[0]
    actual = layer(hidden, hashes)
    error = actual.float() - expected.float()
    assert error.norm() / expected.float().norm() < 0.01
    assert error.abs().max() < 0.02


def test_tp4_reductions_match_single_rank(tmp_path, monkeypatch):
    reference, _, _, _, path, layout, cfg, _, _, hashes, hidden = make_case(tmp_path)
    single = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch)
    expected = single(hidden, hashes)
    shards = [
        make_layer(reference, path, layout, cfg, 4, rank, monkeypatch)
        for rank in range(4)
    ]
    partials = []
    for layer in shards:
        local = layer.multi_head_embedding
        rows = local(hashes).to(torch.bfloat16).float()
        gamma = torch.stack(
            [norm.weight for norm in layer.lookup_per_head_norms]
        ).float()
        eps = torch.finfo(torch.float32).eps
        x = (
            rows * (rows.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma
        ).flatten(1)
        partials.append([*(proj(x) for proj in layer.key_projs), layer.value_proj(x)])
    totals = [torch.stack([part[i] for part in partials]).sum(0) for i in range(3)]
    for rank, layer in enumerate(shards):
        if rank:
            for proj in [layer.value_proj, *layer.key_projs]:
                assert torch.count_nonzero(proj.bias) == 0
        calls = iter(zip(partials[rank], totals))

        def reduce(value, calls=calls):
            local, total = next(calls)
            torch.testing.assert_close(value, local, rtol=0, atol=0)
            return total

        monkeypatch.setattr(engram, "tensor_model_parallel_all_reduce", reduce)
        actual = layer(hidden, hashes)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        assert next(calls, None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_whole_layer_against_cpu(tmp_path, monkeypatch):
    from vllm.forward_context import override_forward_context

    reference, _, _, _, path, layout, cfg, _, _, hashes, hidden = make_case(tmp_path)
    cpu = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch)
    cfg.device_config.device_type = "cuda"
    gpu = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch).cuda()
    context = SimpleNamespace(
        no_compile_layers=cfg.compilation_config.static_forward_context
    )
    with override_forward_context(context):
        actual = gpu(hidden.cuda(), hashes.cuda()).cpu()
    torch.testing.assert_close(actual, cpu(hidden, hashes), rtol=2e-5, atol=3e-6)
