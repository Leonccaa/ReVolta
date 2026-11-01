# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp cache layout in small, and a deterministic allocation trace.

The trace only uses APIs that predate the partial prefix cache, so the same
function records the reference behaviour on the base tree (the golden values in
``test_partial_prefix_cache_qwen4exp.py``) and the behaviour with the partial
prefix cache switched off.
"""

from types import SimpleNamespace
from typing import Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

# Production geometry scaled down 6.5x: a 256-token global block (1664 in
# production) whose sharded main K/V keeps 128 local slots per DCP2 rank.
BLOCK = 256
DCP = 2
HASH_BLOCK = 16
WINDOW = 320
DRAFT_BLOCK = 128
NUM_BLOCKS = 96

GROUP_NAMES = ("main_kv", "compressed_key", "ring", "gdn", "ple", "drafter")


def qwen4exp_kv_cache_config(
    num_speculative_blocks: int = 0,
    num_blocks: int = NUM_BLOCKS,
    num_replayed_speculative_blocks: int = 0,
) -> KVCacheConfig:
    """Groups as the CSA+linear allocator emits them for Flash-Next."""
    main_kv = FullAttentionSpec(
        block_size=BLOCK // DCP,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        dcp_sharded=True,
    )
    compressed = MLAAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        dcp_sharded=False,
    )
    ring = CircularBufferSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.float16,
        dcp_sharded=False,
    )
    gdn = MambaSpec(
        block_size=BLOCK,
        shapes=((4, 64),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_replayed_speculative_blocks=num_replayed_speculative_blocks,
    )
    ple = MambaSpec(
        block_size=BLOCK,
        shapes=((2, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_replayed_speculative_blocks=num_replayed_speculative_blocks,
        tp_replicated=True,
    )
    drafter = SlidingWindowSpec(
        block_size=DRAFT_BLOCK,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=WINDOW,
        dcp_sharded=False,
    )
    specs = (main_kv, compressed, ring, gdn, ple, drafter)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([name], spec) for name, spec in zip(GROUP_NAMES, specs)
        ],
    )


_NONE_HASH_READY = False


def ensure_none_hash() -> None:
    """Initialize the hash seed once: it is random unless PYTHONHASHSEED is
    set, so re-initializing between requests would change their hashes."""
    global _NONE_HASH_READY
    if not _NONE_HASH_READY:
        init_none_hash(sha256)
        _NONE_HASH_READY = True


def make_manager(**kwargs) -> KVCacheManager:
    ensure_none_hash()
    config_kwargs = {
        k: kwargs.pop(k)
        for k in (
            "num_speculative_blocks",
            "num_blocks",
            "num_replayed_speculative_blocks",
        )
        if k in kwargs
    }
    return KVCacheManager(
        qwen4exp_kv_cache_config(**config_kwargs),
        max_model_len=8192,
        hash_block_size=HASH_BLOCK,
        enable_caching=True,
        dcp_world_size=DCP,
        prefix_cache_retention_interval=0,
        **kwargs,
    )


def make_request(
    request_id: str,
    prompt: list[int],
    boundaries: Any = None,
    reasoning_ended: bool | None = None,
) -> Request:
    ensure_none_hash()
    extra = {"prefix_cache_boundaries": boundaries} if boundaries else None
    params = SamplingParams(max_tokens=64, extra_args=extra)
    params.update_from_generation_config({}, eos_token_id=100_000)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        reasoning_ended=reasoning_ended,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def run_chunk(manager, request, num_tokens, hit_tokens=0, hit_blocks=None):
    blocks = manager.allocate_slots(
        request,
        num_tokens,
        num_new_computed_tokens=hit_tokens,
        new_computed_blocks=hit_blocks,
    )
    assert blocks is not None
    request.num_computed_tokens += hit_tokens + num_tokens
    manager.new_step_starts()
    return blocks


def _ids(manager, request_id):
    return [list(ids) for ids in manager.get_block_ids(request_id)]


def allocation_trace(manager) -> list:
    """Two chat turns and a resend, chunked as the scheduler chunks them."""
    out: list = []
    prompt = [(7 * i + 3) % 1000 for i in range(700)]
    first = make_request("first", prompt, boundaries=[600])
    for chunk in (256, 256, 188):
        run_chunk(manager, first, chunk)
        out.append(("first", _ids(manager, "first"), manager.take_new_block_ids()))
    for token in (11, 12, 13):
        first.append_output_token_ids(token)
        run_chunk(manager, first, 1)
    out.append(("first-decode", _ids(manager, "first"), manager.take_new_block_ids()))
    manager.free(first)
    out.append(("free", manager.block_pool.get_num_free_blocks()))

    second_prompt = prompt[:640] + [(5 * i + 1) % 997 for i in range(300)]
    second = make_request("second", second_prompt, boundaries=[900])
    hit_blocks, hit = manager.get_computed_blocks(second)
    out.append(("second-hit", hit, [list(b) for b in hit_blocks.get_block_ids()]))
    remaining = len(second_prompt) - hit
    first_chunk = min(remaining, BLOCK - hit % BLOCK if hit % BLOCK else BLOCK)
    run_chunk(manager, second, first_chunk, hit, hit_blocks)
    out.append(("second", _ids(manager, "second"), manager.take_new_block_ids()))
    while second.num_computed_tokens < len(second_prompt):
        chunk = min(BLOCK, len(second_prompt) - second.num_computed_tokens)
        run_chunk(manager, second, chunk)
    out.append(("second-end", _ids(manager, "second"), manager.take_new_block_ids()))

    third = make_request("third", list(prompt), boundaries=[600])
    hit_blocks, hit = manager.get_computed_blocks(third)
    out.append(("third-hit", hit, [list(b) for b in hit_blocks.get_block_ids()]))
    manager.free(second)
    out.append(("free", manager.block_pool.get_num_free_blocks()))
    return out


def split_trace(partial_cache=None, stops=lambda request, start, end: ()) -> list:
    """``Scheduler._mamba_block_aligned_split`` over a grid of chunk inputs."""
    owner = SimpleNamespace(
        mamba_state_block_size=BLOCK,
        cache_config=SimpleNamespace(block_size=BLOCK),
        use_eagle=False,
        mamba_state_retention_interval=0,
        mamba_dense_boundaries_on_contention=False,
        running=[],
        waiting=[],
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(
                get_replay_boundaries=lambda request, block_size: []
            ),
            partial_cache=partial_cache,
            partial_prefix_stops=stops,
        ),
    )
    out = []
    for num_prompt in (300, 700, 1000):
        for computed in (0, 256, 512, 576):
            for new in (64, 300, 600):
                if computed >= num_prompt:
                    continue
                request = SimpleNamespace(
                    num_computed_tokens=computed,
                    num_prompt_tokens=num_prompt,
                    num_tokens=num_prompt,
                    shared_prefix_boundary=0,
                    prefix_cache_boundaries=(576, 640),
                )
                out.append(Scheduler._mamba_block_aligned_split(owner, request, new))
    return out
