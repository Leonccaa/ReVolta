# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.hy_v4_flash.sm70.dcp import (
    TOPK,
    apply_attn_sink,
    pack_dcp_topk_candidates,
    stable_topk_from_gathered_candidates_triton,
)
from vllm.models.hy_v4_flash.sm70.indexpool import expand_groups, select_tokens
from vllm.platforms import current_platform
from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize("world", [2, 4])
def test_owner_and_topk_oracle(world):
    # Negative and positive scores without ties, then a tie at the cutoff.
    ids = torch.arange(2304)
    scores = ((ids * 47) % ids.numel() - 512).float()
    positions = torch.tensor([2048, 4098, 8195])
    logits = scores.expand(len(positions), -1).clone()
    logits.masked_fill_(ids >= positions[:, None] // 4, -torch.inf)
    for tied in (False, True):
        if tied:
            logits[2, : positions[2] // 4] = 0
        expected = logits.argsort(descending=True, stable=True)[:, : TOPK - 1]
        local = logits.reshape(len(positions), -1, world)
        candidates = local.argsort(dim=1, descending=True, stable=True)[:, :TOPK]
        global_ids = (candidates * world + torch.arange(world)).flatten(1)
        ordered_ids = global_ids.sort().values
        candidate_scores = logits.gather(1, ordered_ids)
        merged = ordered_ids.gather(
            1, candidate_scores.argsort(descending=True, stable=True)[:, : TOPK - 1]
        )
        torch.testing.assert_close(merged, expected)
        tokens = expand_groups(merged.sort().values, positions, 4)
        if tied:
            torch.testing.assert_close(merged[2], torch.arange(TOPK - 1))
        else:
            torch.testing.assert_close(tokens, select_tokens(logits, positions, 4, 511))

        # Reconstruct the source IDs from compact page offsets. Every full
        # group contributes an equal number of tokens to each shard.
        for rank in range(world):
            owned = tokens[:, rank::world]
            local_ids = owned // world
            page, offset = local_ids // (256 // world), local_ids % (256 // world)
            live = owned >= 0
            assert (owned[live] % world == rank).all()
            assert (page[live] == owned[live] // 256).all()
            assert torch.equal((page * 256 + offset * world + rank)[live], owned[live])


@pytest.mark.parametrize("world", [2, 4])
def test_dcp_lse_with_empty_shards(world):
    rng = torch.Generator().manual_seed(72)
    scores = torch.randn(4, 8, 73, generator=rng)
    values = torch.randn(73, 512, generator=rng)
    sinks = torch.linspace(-5, 5, 8)
    mask = torch.arange(73) >= torch.tensor([0, 1, 17, 73])[:, None]
    scores.masked_fill_(mask[:, None], -torch.inf)
    sink_col = sinks.view(1, -1, 1).expand(4, -1, -1)
    expected = torch.cat((scores, sink_col), -1).softmax(-1)[..., :-1] @ values
    outputs, lses = [], []
    for rank in range(world):
        local = scores[..., rank::world]
        lse = local.logsumexp(-1)
        probs = torch.exp(local - lse.nan_to_num(neginf=0).unsqueeze(-1))
        outputs.append(probs @ values[rank::world])
        lses.append(lse)
        assert not outputs[-1][0].any()
        assert lse[0].isneginf().all()
        if rank:
            assert not outputs[-1][1].any()
            assert lse[1].isneginf().all()
    out, lse = _lse_weighted_combine(
        torch.stack(outputs), torch.stack(lses), return_lse=True
    )
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        apply_attn_sink(out, lse, sinks), expected, atol=1e-6, rtol=1e-5
    )
    # The production CUDA path transports FP16 partials with FP32 LSE.
    rounded, _ = _lse_weighted_combine(
        torch.stack(outputs).half().float(), torch.stack(lses), return_lse=True
    )
    rounded = apply_attn_sink(rounded.half(), lse, sinks).float()
    torch.testing.assert_close(rounded, expected, atol=1e-3, rtol=2e-3)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize("world", [2, 4])
def test_cuda_candidate_pack_and_merge(world):
    # Short, empty and long rows, with signed zero ties crossing rank boundaries.
    scores = torch.arange(2304, device="cuda").expand(3, -1).float()
    scores[2, ::2] = -0.0
    scores[2, 1::2] = 0.0
    lengths = torch.tensor([17, 0, 1577], device="cuda")
    scores.masked_fill_(
        torch.arange(2304, device="cuda") >= lengths[:, None], -torch.inf
    )
    packed = []
    for rank in range(world):
        local = scores[:, rank::world].contiguous()
        counts = ((lengths + world - 1 - rank) // world).int()
        indices = torch.empty((3, TOPK), dtype=torch.int32, device="cuda")
        torch.ops._C.qsa_lexicographic_topk(local, counts, indices, TOPK)
        pairs = local.new_empty((3, TOPK, 2))
        pack_dcp_topk_candidates[(3,)](
            local, indices, pairs, local.shape[1], world, rank, TOPK, num_warps=8
        )
        packed.append(pairs)
    actual = torch.empty_like(indices)
    stable_topk_from_gathered_candidates_triton[(3,)](
        torch.cat(packed, dim=1), actual, world, TOPK, num_warps=8
    )
    expected = scores.argsort(descending=True, stable=True)[:, :TOPK].int()
    expected.masked_fill_(torch.arange(TOPK, device="cuda") >= lengths[:, None], -1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_hy4_dcp_eager_config(world):
    from types import SimpleNamespace

    from vllm.config import DeviceConfig, VllmConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.models.config import HYV4FlashForCausalLMConfig

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = SimpleNamespace(enforce_eager=False)
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.decode_context_parallel_size = world
    config.parallel_config.dcp_comm_backend = None
    config.compilation_config.cudagraph_mode = None
    verify = HYV4FlashForCausalLMConfig.verify_and_update_config
    if world == 1:
        verify(config)
        assert (
            config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        )
    else:
        with pytest.raises(NotImplementedError, match="--enforce-eager"):
            verify(config)
        config.model_config.enforce_eager = True
        verify(config)
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        assert config.parallel_config.dcp_comm_backend == "a2a"
        config.parallel_config.dcp_kv_cache_interleave_size = 2
        with pytest.raises(NotImplementedError, match="interleave 1"):
            verify(config)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize("fp8", [False, True])
def test_cuda_shard_paged_and_splitk_lse(fp8):
    from vllm.models.hy_v4_flash.sm70.decode import decode_slots
    from vllm.models.hy_v4_flash.sm70.kernels import (
        _unpack_fp8_ds_mla_torch,
        cache_fp8_ds_mla,
        paged_attention,
    )
    from vllm.models.hy_v4_flash.sm70.splitk import (
        decode_workspace_specs,
        paged_attention_splitk,
    )

    device = torch.device("cuda")
    torch.manual_seed(29)
    world, rank, page = 4, 3, 64
    table = torch.arange(8, -1, -1, dtype=torch.int32, device=device).expand(3, -1)
    length = torch.tensor([1, 256, 2050], dtype=torch.int32, device=device)
    active = torch.zeros(3, dtype=torch.int64, device=device)
    history = torch.arange(511, dtype=torch.int32, device=device).repeat(3, 1)
    slots = torch.empty((3, 512), dtype=torch.int32, device=device)
    counts = torch.empty_like(length)
    decode_slots(
        length, active, table.contiguous(), history, slots, counts, page, 4, world, rank
    )
    assert counts.tolist() == [0, 64, 511]
    owners = (3 + 4 * torch.arange(511, device=device)).int()
    expected_slots = table[2, owners // 256] * page + owners // world % page
    torch.testing.assert_close(slots[2, :511], expected_slots)

    q = torch.randn(3, 32, 576, device=device, dtype=torch.float16)
    kv = torch.randn(9 * page, 576, device=device, dtype=torch.float16)
    if fp8:
        cache = torch.empty((9, page, 656), dtype=torch.uint8, device=device)
        cache_fp8_ds_mla(
            kv[:, :512].contiguous(),
            kv[:, 512:].contiguous(),
            cache,
            torch.arange(9 * page, device=device),
        )
        kv = _unpack_fp8_ds_mla_torch(cache.cpu(), 512).flatten(0, 1).to(device)
    else:
        cache = kv.view(9, page, 576)
    selected = kv[slots.clamp_min(0)].float()
    scale = 0.0625
    logits = q.float() @ selected.transpose(1, 2) * scale
    logits.masked_fill_(slots[:, None] < 0, -torch.inf)
    expected_lse = logits.logsumexp(-1)
    expected = logits.softmax(-1).nan_to_num() @ selected[..., :512]

    lse = q.new_empty(q.shape[:2], dtype=torch.float32)
    partial_state, partial_acc = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in decode_workspace_specs(3, 32, 512, 512)
    ]
    for out in (
        lambda: paged_attention(q, cache, slots, counts, None, scale, 512, lse=lse),
        lambda: paged_attention_splitk(
            q, cache, slots, counts, None, scale, partial_state, partial_acc, lse=lse
        ),
    ):
        torch.testing.assert_close(out().float(), expected, atol=2e-3, rtol=3e-3)
        torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=1e-5)
