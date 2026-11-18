# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.hy_v4_flash.sm70 import indexpool
from vllm.models.hy_v4_flash.sm70.attention import HYV4FlashAttention
from vllm.models.hy_v4_flash.sm70.kernels import index_scores, paged_attention
from vllm.models.hy_v4_flash.sm70.swa import window_indices
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.kv_cache_view import bind_attention_cache
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowMLASpec
from vllm.v1.request import Request
from vllm.v1.worker.utils import compressed_kernel_block_size


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def text_config():
    # Keep the real token limits, use smaller projection dimensions.
    return SimpleNamespace(
        layer_types=["indexed_attention", "sliding_attention"],
        hidden_size=32,
        q_lora_rank=16,
        kv_lora_rank=16,
        num_attention_heads=4,
        swa_num_attention_heads=8,
        qk_nope_head_dim=12,
        swa_qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        swa_qk_rope_head_dim=8,
        v_head_dim=16,
        swa_v_head_dim=16,
        rms_norm_eps=1e-5,
        sliding_window=512,
        compressed_indexer_compress_ratio=4,
        compressed_indexer_group_topk=512,
        compressed_indexer_num_attention_heads=4,
        compressed_indexer_head_dim=16,
        sandwich_norm=True,
        dsa_skip_attn_post_norm=False,
        rope_for_layer=lambda i: {"rope_theta": (1e7, 1e4)[i]},
    )


@pytest.fixture
def vllm_config():
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


def make_layer(
    config, model_config, idx, rows, monkeypatch, world=1, rank=0, device="cpu"
):
    tp = SimpleNamespace(world_size=world, rank_in_group=rank)
    monkeypatch.setattr(parallel_state, "_TP", tp)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.float16),
        torch.device(device),
    ):
        scratch = torch.empty(rows, 2048, dtype=torch.int32)
        return HYV4FlashAttention(
            config, model_config, idx, scratch, f"model.layers.{idx}.self_attn"
        )


def load_layer(layer, weights):
    params = dict(layer.named_parameters())
    loaded = set()
    for name, weight in weights.items():
        name = {
            "gate_proj.weight": "linear_gate.weight",
            "sinks": "learnable_sink_param",
        }.get(name, name.replace("compressed_indexer.", "indexer."))
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, weight)
        loaded.add(name)
    assert loaded == set(params)


@pytest.fixture
def hf_layers():
    hf = pytest.importorskip("transformers.models.hy_v4_flash.modeling_hy_v4_flash")
    cfg = text_config()
    hf_cfg = hf.HYV4FlashTextConfig(
        num_hidden_layers=2,
        mlp_layer_types=["dense", "dense"],
        rope_parameters={
            layer_type: {"rope_type": "default", "mrope_section": [2, 1, 1]}
            for layer_type in cfg.layer_types
        },
        **{
            k: v
            for k, v in vars(cfg).items()
            if k not in {"rope_for_layer", "sandwich_norm", "dsa_skip_attn_post_norm"}
        },
    )
    hf_cfg._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(874)
        layers = [hf.HYV4FlashAttention(hf_cfg, i).eval() for i in range(2)]
        with torch.no_grad():
            for layer in layers:
                layer.sinks.uniform_(-2.0, 3.0)
                if layer.compressed_indexer is not None:
                    layer.compressed_indexer.intra_group_ape.normal_(std=0.3)
    return layers, hf.HYV4FlashRotaryEmbedding(hf_cfg)


class PagedLayer:
    """Feed the actual metadata builder and bind shuffled physical pages."""

    def __init__(self, model, config, length, requests=1):
        self.layer = model
        self.config = config
        self.device = model.topk_indices_buffer.device
        self.owners = [model.attn]
        if model.indexer is not None:
            self.owners += [model.indexer.k_cache, model.indexer.tail_cache]
        self.tables = {}
        for owner in self.owners:
            spec = owner.spec
            ring = isinstance(spec, CircularBufferSpec)
            blocks = 1 if ring else (length + spec.block_size - 1) // spec.block_size
            page = (
                compressed_kernel_block_size(spec)
                if spec.storage_block_size != spec.block_size
                else spec.block_size
            )
            page_ratio = spec.storage_block_size // page
            table = torch.arange(
                1, requests * blocks + 1, device=self.device, dtype=torch.int32
            )
            self.tables[owner.prefix] = table.flip(0).view(requests, blocks)
            shape = owner.backend.get_kv_cache_shape(
                (requests * blocks + 1) * page_ratio, page, 1, spec.head_size
            )
            # Pad the page stride without changing the contiguous layout of each row.
            storage = torch.zeros(
                (shape[0], shape[1] + 1, shape[2]), device=self.device, dtype=spec.dtype
            )
            owner.bind_kv_cache(storage[:, : shape[1]])

    @torch.inference_mode()
    def step(self, *requests):
        # Each entry is (query activations, request prefix length, RoPE offset).
        starts = torch.tensor(
            [0, *(len(x) for x, _, _ in requests)], dtype=torch.int32
        ).cumsum(0)
        lengths = torch.tensor([len(x) + p for x, p, _ in requests], dtype=torch.int32)
        positions = [
            torch.arange(p, p + len(x), device=self.device) for x, p, _ in requests
        ]
        metadata = {}
        for owner in self.owners:
            table = self.tables[owner.prefix]
            span = owner.spec.block_size
            slots = torch.empty(0, dtype=torch.int64, device=self.device)
            if owner is self.layer.attn:
                slots = torch.cat(
                    [
                        table[i, p // span].long() * span + p % span
                        for i, p in enumerate(positions)
                    ]
                )
            common = CommonAttentionMetadata(
                query_start_loc=starts.to(self.device),
                query_start_loc_cpu=starts,
                seq_lens=lengths.to(self.device),
                seq_lens_cpu_upper_bound=lengths,
                num_reqs=len(requests),
                num_actual_tokens=int(starts[-1]),
                max_query_len=max(len(x) for x, _, _ in requests),
                max_seq_len=int(lengths.max()),
                block_table_tensor=table,
                slot_mapping=slots,
            )
            builder = owner.backend.get_builder_cls()(
                owner.spec, [owner.prefix], self.config, self.device
            )
            metadata[owner.prefix] = builder.build(0, common)
            assert metadata[owner.prefix].query_start_loc is common.query_start_loc
        phase = torch.cat(
            [p + offset for p, (_, _, offset) in zip(positions, requests)]
        )
        with set_forward_context(metadata, self.config):
            return self.layer(phase, torch.cat([x for x, _, _ in requests]))

    def logical_indices(self, req, rows):
        layer = self.layer
        table = self.tables[layer.attn.prefix][req]
        ids = layer.topk_indices_buffer[rows, : layer.width].clone()
        page = layer.attn.kv_cache.shape[1]
        matches = (ids[..., None] // page == table).int().argmax(-1)
        return (matches * page + ids % page).masked_fill(ids < 0, -1)


def hf_attention(layer, rotary, x, offset=0, selected=None):
    n = len(x)
    positions = torch.arange(n)
    visible = positions[:, None] >= positions[None, :]
    if layer.attention_type == "sliding_attention":
        visible &= positions[:, None] - positions[None, :] < layer.config.sliding_window
    if selected is not None:
        visible &= selected
    mask = torch.zeros(n, n).masked_fill(~visible, -torch.inf)[None, None]
    phases = (positions + offset).expand(3, 1, n)
    cos_sin = rotary(x.float()[None], phases, layer_type=layer.attention_type)
    if selected is not None:
        with patch.object(layer, "compressed_indexer", None):
            return layer(x.float()[None], cos_sin, mask, position_ids=phases)
    return layer(x.float()[None], cos_sin, mask, position_ids=phases)


def verify_boundary_flips(ref, x, visible, picked, logits, positions):
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
        _apply_partial_rotary_pos_emb,
    )

    indexer = ref.compressed_indexer
    x = x.float()[None]
    raw = indexer.token_key_norm(indexer.token_key_projection(x))
    keys = indexer.pool_groups(raw)
    g = keys.shape[1]
    # Absolute rotary offset of 11 above, while grouping follows logical slots.
    phase = (torch.arange(x.shape[1]) + 11)[None, None]
    kg = indexer.rotary_emb(keys, phase[..., : g * 4 : 4], "indexed_attention")
    keys = _apply_partial_rotary_pos_emb(keys.unsqueeze(2), *kg, indexer.rope_dim, 2)
    hidden = x[:, positions]
    q = indexer.query_projection(ref.q_a_layernorm(ref.q_a_proj(hidden))).view(
        1, -1, indexer.num_heads, indexer.head_dim
    )
    qg = indexer.rotary_emb(q, phase[..., positions], "indexed_attention")
    q = _apply_partial_rotary_pos_emb(q, *qg, indexer.rope_dim, 2)
    w = (
        indexer.head_weight_projection(hidden)
        * (indexer.num_heads * indexer.head_dim) ** -0.5
    )
    scores = ((q[0] @ keys[0, :, 0].T).relu() * w[0, :, :, None]).sum(1)
    for row, pos in enumerate(positions.tolist()):
        count = pos // 4
        expected = visible[pos, : count * 4 : 4]
        actual = picked[pos, : count * 4 : 4]
        errors = (logits[row, :count] - scores[row, :count]).abs()
        assert torch.isfinite(errors).all()
        assert errors.norm() / scores[row, :count].norm() < 5e-3
        if not torch.equal(actual, expected):
            assert actual.sum() == expected.sum() == 511
            assert (
                scores[row, :count][expected & ~actual].max()
                - scores[row, :count][actual & ~expected].min()
                <= 2 * errors.max()
            )


def assert_parity(out, expected):
    assert torch.isfinite(out).all()
    comparison = out.float() - expected.float()
    assert comparison.norm() / expected.float().norm() < 5e-3
    torch.testing.assert_close(out.float(), expected.float(), atol=6e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "idx,length,cuts",
    [
        (0, 2048, [1, 3, 4, 259, 2047]),
        (0, 2081, [3, 4, 101, 256, 2048, 2053, 2080]),
        (1, 519, [3, 511, 512, 513]),
    ],
)
@torch.inference_mode()
def test_layer_against_transformers(
    vllm_config, hf_layers, monkeypatch, idx, length, cuts
):
    layers, rotary = hf_layers
    ref = layers[idx]
    model = make_layer(vllm_config, text_config(), idx, length, monkeypatch)
    load_layer(model, ref.state_dict())
    x = torch.randn(length, 32, generator=torch.Generator().manual_seed(607)).half()
    expected, weights = hf_attention(ref, rotary, x, offset=11)
    whole = PagedLayer(model, vllm_config, length)
    select = indexpool.select_tokens
    with patch.object(indexpool, "select_tokens", wraps=select) as called:
        actual = whole.step((x, 0, 11))
    visible = weights[0, 0] > 0
    picked = whole.logical_indices(0, slice(None)).long()
    # Scatter into an extra column for -1 without treating padding as token 0.
    recovered = torch.zeros(length, length + 1, dtype=torch.bool)
    recovered.scatter_(1, picked.masked_fill(picked < 0, length), True)
    recovered = recovered[:, :length]
    if called.call_count:
        (logits, positions, _, _), _ = called.call_args
        assert torch.equal(recovered[:2048], visible[:2048])
        verify_boundary_flips(ref, x, visible, recovered, logits, positions)
        assert (actual.float() - expected[0]).norm() / expected[0].norm() < 5e-3
        expected, _ = hf_attention(ref, rotary, x, offset=11, selected=recovered)
    else:
        assert torch.equal(recovered, visible)
    assert_parity(actual, expected[0])
    if idx == 0:
        pool = model.indexer.k_cache
        first_group = whole.tables[pool.prefix][0, 0]
        closed = pool.kv_cache[first_group, 0].clone()

    paged = PagedLayer(model, vllm_config, length)
    outputs = []
    start = 0
    for end in [*cuts, length]:
        outputs.append(paged.step((x[start:end], start, 11)))
        assert torch.equal(
            paged.logical_indices(0, slice(0, end - start)), picked[start:end]
        )
        if idx == 0 and end in (3, 4):
            first_group = paged.tables[pool.prefix][0, 0]
            if end == 3:
                assert torch.count_nonzero(pool.kv_cache[first_group, 0]) == 0
            else:
                torch.testing.assert_close(pool.kv_cache[first_group, 0], closed)
        start = end
    assert_parity(torch.cat(outputs), actual)

    # The last eight rows sent as one chunk or as one-token decode steps.
    paged = PagedLayer(model, vllm_config, length)
    paged.step((x[:-8], 0, 11))
    decoded = []
    for p in range(length - 8, length):
        decoded.append(paged.step((x[p : p + 1], p, 11)))
        assert torch.equal(paged.logical_indices(0, slice(0, 1)), picked[p : p + 1])
    assert_parity(torch.cat(decoded), actual[-8:])


@torch.inference_mode()
@pytest.mark.parametrize("idx", [0, 1])
def test_tp4_weight_loading_and_output(vllm_config, hf_layers, monkeypatch, idx):
    weights = hf_layers[0][idx].state_dict()
    length = 11
    x = torch.randn(length, 32, generator=torch.Generator().manual_seed(67)).half()
    model = make_layer(vllm_config, text_config(), idx, length, monkeypatch)
    load_layer(model, weights)
    results, head_results, expected_heads = [], [], []
    handle = model.o_proj.register_forward_pre_hook(
        lambda _, args: expected_heads.append(args[0])
    )
    full = PagedLayer(model, vllm_config, length).step((x, 0, 0))
    handle.remove()
    for rank in range(4):
        shard = make_layer(
            vllm_config, text_config(), idx, length, monkeypatch, world=4, rank=rank
        )
        load_layer(shard, weights)
        shard.o_proj.reduce_results = False
        shard.post_proj_norm = None
        handle = shard.o_proj.register_forward_pre_hook(
            lambda _, args: head_results.append(args[0])
        )
        results.append(PagedLayer(shard, vllm_config, length).step((x, 0, 0)))
        handle.remove()
    torch.testing.assert_close(
        torch.cat(head_results, dim=-1), expected_heads[0], atol=0, rtol=0
    )
    reduced = torch.stack(results).float().sum(0).half()
    assert_parity(model.post_proj_norm(reduced), full)


@torch.inference_mode()
def test_mixed_requests_preserve_independent_group_tails(
    vllm_config, hf_layers, monkeypatch
):
    length = 21
    model = make_layer(vllm_config, text_config(), 0, length * 2, monkeypatch)
    load_layer(model, hf_layers[0][0].state_dict())
    rng = torch.Generator().manual_seed(45)
    a, b = torch.randn(2, length, 32, generator=rng).half().unbind()
    pool = model.indexer.k_cache

    def closed_groups(runner, req):
        return pool.kv_cache[runner.tables[pool.prefix][req, 0], : length // 4]

    solo = PagedLayer(model, vllm_config, length)
    expected_a = solo.step((a, 0, 17))
    pool_a = closed_groups(solo, 0).clone()
    solo = PagedLayer(model, vllm_config, length)
    expected_b = solo.step((b, 0, 223))
    pool_b = closed_groups(solo, 0).clone()
    paged = PagedLayer(model, vllm_config, length, requests=2)
    for a0, a1, b0, b1 in [(0, 3, 0, 5), (3, 4, 5, 12), (4, 21, 12, 21)]:
        result = paged.step((a[a0:a1], a0, 17), (b[b0:b1], b0, 223))
        assert_parity(result[: a1 - a0], expected_a[a0:a1])
        assert_parity(result[a1 - a0 :], expected_b[b0:b1])
    torch.testing.assert_close(closed_groups(paged, 0), pool_a, atol=0, rtol=0)
    torch.testing.assert_close(closed_groups(paged, 1), pool_b, atol=0, rtol=0)


def test_window_boundaries():
    rows = window_indices(torch.tensor([510, 511, 512]), 512)
    assert rows[0].tolist() == [*range(511), -1]
    assert rows[1].tolist() == list(range(512))
    assert rows[2].tolist() == list(range(1, 513))


@pytest.mark.parametrize("n_swa", [0, 30])
def test_scheduler_page_layout(vllm_config, monkeypatch, n_swa):
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    vllm_config.model_config = SimpleNamespace(max_model_len=4096)
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    cfg = text_config()
    cfg.kv_lora_rank = 512
    cfg.qk_rope_head_dim = cfg.swa_qk_rope_head_dim = 64
    cfg.compressed_indexer_head_dim = 128
    cfg.layer_types = ["indexed_attention"] * 12 + ["sliding_attention"] * n_swa
    cfg.rope_for_layer = lambda i: {"rope_theta": 1e7 if i < 12 else 1e4}
    for idx in range(12 + n_swa):
        make_layer(vllm_config, cfg, idx, 1, monkeypatch)
    layers = vllm_config.compilation_config.static_forward_context
    groups = get_kv_cache_groups(
        vllm_config, {name: layer.spec for name, layer in layers.items()}
    )
    cache = get_kv_cache_config_from_groups(vllm_config, groups, 128 << 20)
    coordinator = HybridKVCacheCoordinator(
        generate_scheduler_kv_cache_config([cache]),
        4096,
        2048,
        False,
        True,
        False,
        1,
        1,
        256,
    )
    assert len(set(name for group in groups for name in group.layer_names)) == len(
        layers
    )
    assert (
        sum(isinstance(layer.spec, SlidingWindowMLASpec) for layer in layers.values())
        == n_swa
    )
    ring_groups = [
        i for i, group in enumerate(groups) if not group.kv_cache_spec.prefix_cacheable
    ]
    assert len(ring_groups) == 1
    ring = coordinator.single_type_managers[ring_groups[0]]
    assert len(ring.allocate_new_blocks("request", 2053, 2053)) == 1
    assert ring.allocate_new_blocks("request", 2081, 2081) == []
    # Allocate just two real allocator strides, then check each owner can bind.
    for tensor in cache.kv_cache_tensors:
        stride = tensor.size // cache.num_blocks
        raw = torch.empty(stride * 2, dtype=torch.int8)
        for name in tensor.shared_by:
            layer = layers[name]
            spec = layer.spec
            page = (
                compressed_kernel_block_size(spec)
                if spec.storage_block_size != spec.block_size
                else spec.block_size
            )
            shape = layer.backend.get_kv_cache_shape(
                2 * spec.storage_block_size // page, page, 1, spec.head_size
            )
            layer.bind_kv_cache(
                bind_attention_cache(
                    raw,
                    shape=shape,
                    dtype=spec.dtype,
                    page_size_bytes=spec.page_size_bytes,
                    scheduler_block_size=spec.storage_block_size,
                    kernel_block_size=page,
                    block_dim=0,
                )
            )
            assert layer.kv_cache.shape == shape
            assert layer.kv_cache.stride(0) * layer.kv_cache.element_size() == stride


@torch.inference_mode()
def test_page_aligned_prefix_reuse(vllm_config, hf_layers, monkeypatch):
    prefix, length = 2048, 2061
    cfg = text_config()
    x = torch.randn(length, 32, generator=torch.Generator().manual_seed(92)).half()
    vllm_config.model_config = SimpleNamespace(max_model_len=4096)
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    models = [make_layer(vllm_config, cfg, i, length, monkeypatch) for i in range(2)]
    for i, model in enumerate(models):
        load_layer(model, hf_layers[0][i].state_dict())
    # Compare against the same split without a cache hit, so the batch shapes match.
    expected = []
    for model in models:
        cold = PagedLayer(model, vllm_config, length)
        cold.step((x[:prefix], 0, 0))
        expected.append(cold.step((x[prefix:], prefix, 0)))
    owners = vllm_config.compilation_config.static_forward_context
    groups = get_kv_cache_groups(
        vllm_config, {name: layer.spec for name, layer in owners.items()}
    )
    cache = get_kv_cache_config_from_groups(vllm_config, groups, 2 << 20)
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([cache]),
        max_model_len=4096,
        max_in_flight_tokens=2048,
        hash_block_size=256,
        enable_caching=True,
    )
    paged = [PagedLayer(model, vllm_config, length) for model in models]
    for tensor in cache.kv_cache_tensors:
        raw = torch.zeros(tensor.size, dtype=torch.int8)
        for name in tensor.shared_by:
            owner = owners[name]
            spec = owner.spec
            page = (
                compressed_kernel_block_size(spec)
                if spec.storage_block_size != spec.block_size
                else spec.block_size
            )
            shape = (
                cache.num_blocks * spec.storage_block_size // page,
                page,
                spec.head_size,
            )
            owner.bind_kv_cache(
                bind_attention_cache(
                    raw,
                    shape=shape,
                    dtype=spec.dtype,
                    page_size_bytes=spec.page_size_bytes,
                    scheduler_block_size=spec.storage_block_size,
                    kernel_block_size=page,
                    block_dim=0,
                )
            )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(256, sha256)
    first = Request(
        "first",
        list(range(prefix)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=hasher,
    )
    assert manager.allocate_slots(first, prefix) is not None
    group_by_owner = {
        name: i for i, group in enumerate(groups) for name in group.layer_names
    }

    def set_tables(request, runner):
        block_ids = manager.get_block_ids(request.request_id)
        for owner in runner.owners:
            runner.tables[owner.prefix] = torch.tensor(
                [block_ids[group_by_owner[owner.prefix]]], dtype=torch.int32
            )

    for runner in paged:
        set_tables(first, runner)
        runner.step((x[:prefix], 0, 0))
    manager.free(first)
    second = Request(
        "second",
        list(range(length)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=hasher,
    )
    blocks, hit = manager.get_computed_blocks(second)
    assert hit == prefix
    ring_gid = group_by_owner[models[0].indexer.tail_cache.prefix]
    assert not blocks.blocks[ring_gid]
    assert manager.allocate_slots(second, length - hit, hit, blocks) is not None
    for runner, target in zip(paged, expected):
        set_tables(second, runner)
        result = runner.step((x[prefix:], prefix, 0))
        torch.testing.assert_close(result, target, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize("heads,width", [(8, 2048), (16, 512)])
def test_cuda_latent_aggregation(heads, width):
    from vllm import _custom_ops as ops
    from vllm.models.hy_v4_flash.sm70.kernels import paged_attention_torch

    torch.manual_seed(23)
    q = torch.randn(4, heads, 576, device="cuda").half()
    # Include an unscaled QK logit beyond FP16 range and non-zero sinks.
    q[0, :, :512].fill_(40)
    latent = torch.randn(width, 512, device="cuda").half()
    latent[0].fill_(40)
    rotary = torch.randn(width, 64, device="cuda").half()
    sinks = torch.linspace(-2, 5, heads, device="cuda")
    slots = torch.arange(width, device="cuda").reshape(-1, 64).flip(0).flatten()
    cache = torch.zeros(width // 64, 65, 576, device="cuda", dtype=torch.float16)[
        :, :64
    ]
    ops.concat_and_cache_mla(latent, rotary, cache, slots, "auto", sinks[:1])
    torch.testing.assert_close(
        cache[slots // 64, slots % 64], torch.cat((latent, rotary), -1), atol=0, rtol=0
    )
    indices = slots.int().expand(4, -1).contiguous()
    counts = torch.tensor([1, width, width - 3, 511], dtype=torch.int32, device="cuda")
    expected = paged_attention_torch(
        q.cpu(), cache.cpu(), indices.cpu(), counts.cpu(), sinks.cpu(), 1 / 16, 512
    )
    actual = paged_attention(q, cache, indices, counts, sinks, 1 / 16, 512)
    assert_parity(actual.cpu(), expected)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
def test_cuda_weighted_pool_scores():
    torch.manual_seed(41)
    query = torch.randn(7, 16, 128, device="cuda").half()
    weights = torch.randn(7, 16, device="cuda") / 32
    pool = torch.randn(9, 65, 128, device="cuda").half()[:, :64]
    table = torch.tensor([8, 2, 6, 0, 4, 3, 1, 5, 7], device="cuda", dtype=torch.int32)
    groups = torch.arange(522)
    keys = pool.cpu()[table.cpu()[groups // 64], groups % 64].float()
    expected = ((query.cpu().float() @ keys.T).relu() * weights.cpu()[..., None]).sum(1)
    expected.masked_fill_(groups >= (2054 + torch.arange(7)[:, None]) // 4, -torch.inf)
    actual = index_scores(query, weights, pool, table, 2054, 522, 4, 1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=1e-4)
