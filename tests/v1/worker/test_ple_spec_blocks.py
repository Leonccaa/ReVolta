# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE state allocation, metadata, and mixed-group copy context."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as E
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_conv_copy_spec,
    get_mamba_groups,
    get_temporal_copy_spec,
)


def spec(n, ple=True):
    return MambaSpec(
        block_size=16,
        shapes=((13, 8),) if ple else ((7, 8), (16,)),
        dtypes=(torch.float16,) if ple else (torch.float16, torch.float32),
        mamba_type=E.SHORT_CONV if ple else E.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=n,
        tp_replicated=ple,
    )


def builder(n, full):
    cfg = NS(
        num_speculative_tokens=4,
        parallel_config=NS(decode_context_parallel_size=1),
        speculative_config=NS(num_speculative_tokens=4, parallel_drafting=False),
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.FULL if full else CUDAGraphMode.NONE,
            max_cudagraph_capture_size=20,
        ),
        scheduler_config=NS(max_num_seqs=4),
        cache_config=NS(mamba_cache_mode="align"),
    )
    return PleShortConvAttentionMetadataBuilder(
        spec(n), ["ple"], cfg, torch.device("cpu")
    )


def common(lengths):
    starts = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int()
    n = len(lengths)
    # Includes a boundary crossing: the first request's running column is 2.
    seq = torch.tensor([33, *([18] * (n - 1))], dtype=torch.int32)
    table = torch.arange(1, n * 10 + 1, dtype=torch.int32).reshape(n, 10)
    return CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=seq,
        num_reqs=n,
        num_actual_tokens=sum(lengths),
        max_query_len=max(lengths),
        max_seq_len=int(seq.max()),
        block_table_tensor=table,
        slot_mapping=torch.zeros(sum(lengths), dtype=torch.int64),
        causal=True,
        is_prefilling=torch.tensor([q > 1 for q in lengths]),
        _seq_lens_cpu=seq,
        seq_lens_cpu_upper_bound=seq,
    )


@pytest.mark.parametrize(
    "full,lengths,drafts",
    [
        (False, [5, 1, 3], [4, -1, -1]),
        (True, [5, 5, 0], [4, 4, -1]),
        (True, [1, 1], [-1, -1]),
    ],
)
def test_ple_zero_matches_four_metadata(full, lengths, drafts):
    b4, b0 = builder(4, full), builder(0, full)
    m = common(lengths)
    args = dict(
        num_accepted_tokens=torch.tensor([2, 1, 1][: len(lengths)]),
        num_decode_draft_tokens_cpu=torch.tensor(drafts),
    )
    old, new = b4.build(0, m, **args), b0.build(0, m, **args)
    for name in (
        "state_indices_tensor",
        "spec_state_indices_tensor",
        "num_accepted_tokens",
        "has_initial_states_d",
        "has_initial_states_p",
        "spec_query_start_loc",
    ):
        a, b = getattr(old, name), getattr(new, name)
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if full and drafts[0] >= 0:
        assert (
            new.spec_state_indices_tensor.data_ptr()
            == b0.spec_state_indices_tensor.data_ptr()
        )
        # Rebuild to prove persistent pointer reuse, rather than mere value equality.
        again = b0.build(0, m, **args)
        assert (
            again.spec_state_indices_tensor.data_ptr()
            == new.spec_state_indices_tensor.data_ptr()
        )
    # Base non-spec FULL graph copy broadcasts one gathered column into five.
    if full and drafts[0] < 0:
        assert new.state_indices_tensor_d.shape == (2, 5)
        assert torch.equal(
            new.state_indices_tensor_d, new.state_indices_tensor[:, None].expand(-1, 5)
        )


def test_zero_spec_group_context_and_stride():
    cfg = KVCacheConfig(
        20,
        [],
        [KVCacheGroupSpec(["gdn"], spec(4, False)), KVCacheGroupSpec(["ple"], spec(0))],
    )
    # GDN keeps speculative blocks while PLE uses its running block.
    get_mamba_groups(cfg)
    funcs = {
        E.GDN_ATTN: (get_conv_copy_spec, get_temporal_copy_spec),
        E.SHORT_CONV: (get_conv_copy_spec,),
    }
    ctx = MambaSpecDecodeGPUContext.create(
        2, cfg, funcs, torch.device("cpu"), lambda n, dtype: torch.zeros(n, dtype=dtype)
    )
    forward = {
        "gdn": NS(kv_cache=[torch.zeros(20, 7, 8), torch.zeros(20, 16)]),
        "ple": NS(kv_cache=[torch.zeros(20, 13, 8)]),
    }
    # MRv2's per-group table widths: cdiv(max_len, B) + spec blocks.
    with pytest.raises(AssertionError, match="share stride"):
        ctx.initialize_from_forward_context(
            cfg,
            forward,
            funcs,
            [
                torch.zeros(2, 12, dtype=torch.int32),
                torch.zeros(2, 8, dtype=torch.int32),
            ],
        )
    # Equal physical table padding preserves the shared-stride copy contract.
    ctx.initialize_from_forward_context(
        cfg,
        forward,
        funcs,
        [torch.zeros(2, 12, dtype=torch.int32), torch.zeros(2, 12, dtype=torch.int32)],
    )
    assert ctx.is_initialized and ctx.block_table_stride_req == 12
    assert ctx.state_conv_widths.tolist() == [7, 0, 13]


@pytest.mark.parametrize("mode", ["align", "none", "all"])
def test_ple_spec_retains_in_block_history(mode):
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    layer.model_config = NS(dtype=torch.float16)
    layer.cache_config = NS(mamba_cache_dtype="auto")
    layer.hc_hidden_size = 8
    layer.conv_state_len = 9
    layer.num_spec_tokens = 4
    cfg = NS(
        cache_config=NS(
            mamba_block_size=16, mamba_page_size_padded=None, mamba_cache_mode=mode
        ),
        speculative_config=NS(num_speculative_state_tokens=lambda: 4),
    )
    old = MambaBase.get_kv_cache_spec(layer, cfg)
    new = layer.get_kv_cache_spec(cfg)
    assert new.num_speculative_blocks == 0
    assert new == replace(old, num_speculative_blocks=0)
    assert new.shapes == old.shapes
    assert new.page_size_bytes == old.page_size_bytes


def test_group_padding_preserves_attention_and_logical_specs():
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker import mamba_utils

    cfg = KVCacheConfig(
        20,
        [],
        [
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
                ),
            ),
            KVCacheGroupSpec(["ple"], spec(0)),
            KVCacheGroupSpec(["gdn"], spec(4, False)),
        ],
    )
    widths = [65, 8, 12]
    pad = getattr(mamba_utils, "pad_mamba_block_table_widths", None)
    assert pad is not None, "runner needs equal physical Mamba table strides"
    assert pad(cfg, widths) == [65, 12, 12]
    assert widths == [65, 8, 12]
    assert cfg.kv_cache_groups[1].kv_cache_spec.num_speculative_blocks == 0
    ids, representative = get_mamba_groups(cfg)
    assert ids == [1, 2] and representative.num_speculative_blocks == 4


def test_other_mamba_spec_counts_still_must_agree():
    cfg = KVCacheConfig(
        20,
        [],
        [
            KVCacheGroupSpec(["a"], spec(4, False)),
            KVCacheGroupSpec(["b"], spec(2, False)),
            KVCacheGroupSpec(["ple"], spec(0)),
        ],
    )
    with pytest.raises(AssertionError, match="num_speculative_blocks"):
        get_mamba_groups(cfg)


def test_prefix_hits_and_host_state_demand_do_not_depend_on_ple_spec_blocks():
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_offload.cpu.spec import mamba_state_slots
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hits = []
    for count in (4, 0):
        cfg = KVCacheConfig(
            1000,
            [],
            [
                KVCacheGroupSpec(
                    ["attn"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
                    ),
                ),
                KVCacheGroupSpec(["gdn"], spec(4, False)),
                KVCacheGroupSpec(["ple"], spec(count)),
            ],
        )
        manager = KVCacheManager(
            cfg, 4096, 16, use_eagle=True, prefix_cache_retention_interval=0
        )
        prompt = list(range(100, 200))

        def request(name, tokens):
            return Request(
                request_id=name,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(max_tokens=16),
                pooling_params=None,
                block_hasher=get_request_block_hasher(16, sha256),
            )

        req = request("first", prompt + [2001])
        while req.num_computed_tokens < req.num_tokens:
            manager.new_step_starts()
            start = req.num_computed_tokens
            end = min((start // 16 + 1) * 16, req.num_tokens)
            assert manager.allocate_slots(req, end - start, num_lookahead_tokens=4)
            req.num_computed_tokens = end
        manager.free(req)
        _, hit = manager.get_computed_blocks(request("sibling", prompt + [3001]))
        hits.append(hit)
        assert mamba_state_slots(512, 16, 16, spec(count), 0, 4096) == (
            mamba_state_slots(512, 16, 16, spec(4), 0, 4096)
        )
    assert hits[0] == hits[1] and hits[0] > 0
