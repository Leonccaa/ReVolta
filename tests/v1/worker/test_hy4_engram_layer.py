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


def make_case(tmp_path, *, model_shape=False):
    torch.manual_seed(77)
    sizes = (
        [[11, 13, 17, 19, 23, 29, 31, 37], [41, 43, 47, 53, 59, 61, 67, 71]]
        if model_shape
        else [[11, 13], [17, 19]]
    )
    head_dim = 96 if model_shape else 4
    config = SimpleNamespace(
        hidden_size=3072 if model_shape else 8,
        hc_mult=4 if model_shape else 2,
        engram_layer_ids=[2],
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=len(sizes[0]),
        engram_n_embed_per_ngram=len(sizes[0]) * head_dim,
        engram_fp8_embedding=True,
        engram_pad_id=3,
        engram_seed=42,
        engram_layer_head_vocab_sizes={"2": sizes},
        engram_lookup_table_size=32,
        vocab_size=32,
    )
    reference = HYV4FlashEngramLayer(config, 2)
    source = torch.randn(sum(map(sum, sizes)), head_dim, dtype=torch.bfloat16)
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


def checkpoint_reference(reference, source, hidden, ngrams, pad):
    # Match the checkpoint's offline BF16 recipe; use FP32 from lookup onward.
    reference.half().float()
    reference.fp8_embedding = False
    reference.multi_head_embedding.embedding.weight.copy_(
        _fp8_round_trip(source).float()
    )
    streams = hidden.float()
    return streams + reference(streams[None], ngrams[None], (~pad)[None])[0]


def output_errors(actual, expected):
    error = actual.float() - expected
    return float(error.norm() / expected.norm()), float(error.abs().max())


@torch.inference_mode()
def test_fp16_model_shape_against_transformers(tmp_path, monkeypatch, record_property):
    reference, source, _, _, path, layout, cfg, tokens, ngrams, hashes, hidden = (
        make_case(tmp_path, model_shape=True)
    )
    layer = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch).half()
    hidden = hidden.half()
    pad = tokens == layout.pad_token_id
    expected = checkpoint_reference(reference, source, hidden, ngrams, pad)
    actual = layer(hidden, hashes)
    rel_l2, max_abs = output_errors(actual, expected)
    record_property("native_rel_l2", rel_l2)
    record_property("native_max_abs", max_abs)
    assert rel_l2 < 0.01
    assert max_abs < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_tokens", [1, 6])
@torch.inference_mode()
def test_cuda_fused_engram_against_transformers(
    tmp_path, monkeypatch, num_tokens, record_property
):
    from vllm.forward_context import override_forward_context

    reference, source, _, _, path, layout, cfg, tokens, ngrams, hashes, hidden = (
        make_case(tmp_path, model_shape=True)
    )
    cfg.device_config.device_type = "cuda"
    gpu = make_layer(reference, path, layout, cfg, 1, 0, monkeypatch).cuda().half()
    tokens, ngrams = tokens[:num_tokens], ngrams[:num_tokens]
    hidden = hidden[:num_tokens].half()
    pad = tokens == layout.pad_token_id
    expected = checkpoint_reference(reference, source, hidden, ngrams, pad)
    context = SimpleNamespace(
        no_compile_layers=cfg.compilation_config.static_forward_context
    )
    streams = hidden.cuda()
    ids = hashes[:num_tokens].to(device="cuda", dtype=torch.int32)
    with override_forward_context(context):
        native = gpu.forward_native(streams, ids).cpu()
        fused = gpu(streams, ids).cpu()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gpu(streams, ids)
        # The graph must read the updated hash contents, not capture a pad decision.
        ids[0].fill_(-1)
        graph.replay()
        torch.testing.assert_close(captured, gpu(streams, ids), rtol=0, atol=0)
        torch.testing.assert_close(captured[0].cpu(), hidden[0], rtol=0, atol=0)
    torch.testing.assert_close(fused[pad], hidden[pad], rtol=0, atol=0)

    for name, before, after in zip(
        ("rel_l2", "max_abs"),
        output_errors(native, expected),
        output_errors(fused, expected),
    ):
        record_property(f"native_{name}", before)
        record_property(f"fused_{name}", after)
        assert after <= 10 * before, f"{name}: native={before}, fused={after}"
