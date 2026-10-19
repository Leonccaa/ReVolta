# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hybrid DCP/MTP layout and MRv2 hooks work without a KV connector."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu import kv_connector


@pytest.mark.parametrize("prefix_cache", [True, False])
def test_tp4_dcp2_mtp4_layout_without_offload(prefix_cache):
    config = SimpleNamespace(
        kv_transfer_config=None,
        cache_config=SimpleNamespace(
            cache_dtype="fp8_e4m3",
            block_size=64,
            hash_block_size=None,
            enable_prefix_caching=prefix_cache,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
        ),
    )
    attention = FullAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=256, dtype=torch.uint8
    )
    gdn = MambaSpec(
        block_size=128,
        shapes=((16, 128, 128),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=4,
    )
    ple = MambaSpec(
        block_size=128,
        shapes=((4, 2560),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=4,
        tp_replicated=True,
    )
    cache = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target.attn", "draft.attn"], attention),
            KVCacheGroupSpec(["target.gdn", "draft.gdn"], gdn),
            KVCacheGroupSpec(["target.ple", "draft.ple"], ple),
        ],
    )
    assert resolve_kv_cache_block_sizes(cache, config) == (128, 128)
    assert gdn.max_memory_usage_bytes(config) == 6 * gdn.page_size_bytes
    assert ple.max_memory_usage_bytes(config) == 6 * ple.page_size_bytes


def test_mrv2_without_offload_never_resolves_active_connector(monkeypatch):
    monkeypatch.setattr(kv_connector, "has_kv_transfer_group", lambda: False)

    def forbidden():
        raise AssertionError("offload group must not be used")

    monkeypatch.setattr(kv_connector, "get_kv_transfer_group", forbidden)
    connector = kv_connector.get_kv_connector(
        SimpleNamespace(kv_transfer_config=None), {}
    )
    step = SimpleNamespace(kv_connector_metadata=None)
    connector.handle_preemptions(step)
    connector.pre_forward(step)
    assert connector.post_forward(set()) is None
    assert connector.no_forward(step) is kv_connector.EMPTY_MODEL_RUNNER_OUTPUT
