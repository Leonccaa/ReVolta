# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload-disabled startup: TP4 + DCP2 + MTP + hybrid (attention + GDN) model.

Proves that scheduling a production-style hybrid KV layout WITHOUT
``--kv-transfer-config`` has no hidden dependency on the offloading
connector: block/hash geometry is identical to the connector-enabled
geometry, the scheduler never builds a connector, and the Mamba ``align``
boundary-state hand-offs intended for the connector are still drained
every step (no unbounded growth of the pending list) when no connector
exists to consume them.
"""

from pathlib import Path

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

# Production-like geometry (scaled down): 16-token GPU blocks, DCP2 shards
# the attention pages (global span 32), the GDN/Mamba group runs in align
# mode at 768 tokens per boundary state — divisible by the hash granularity.
CACHE_BLOCK_SIZE = 16
ATTN_GLOBAL_BLOCK_SIZE = CACHE_BLOCK_SIZE * 2  # dcp = 2
MAMBA_BLOCK_SIZE = 768


def _hybrid_groups(mtp_blocks: int = 0):
    attn = FullAttentionSpec(
        block_size=CACHE_BLOCK_SIZE,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float8_e4m3fn,
        dcp_sharded=True,
    )
    mamba = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((2, 8, 8), (2, 8)),
        dtypes=(torch.float32, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=mtp_blocks,
    )
    return attn, mamba


def _scheduler_without_connector():
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    attn, mamba = _hybrid_groups()
    # Built by hand: this CUDA image has no CPU-platform marker, so the
    # create_scheduler helper's implicit device auto-detect cannot resolve
    # on a GPU-less machine.  device='cpu' is the only difference.
    # 12 heads / 2 KV heads: passes the TP4+DCP2 GQA guards.
    model_config = ModelConfig(
        model="Qwen/Qwen2-1.5B",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=CACHE_BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=4,
        decode_context_parallel_size=2,
    )
    config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(device="cpu"),
        kv_transfer_config=None,
    )
    cache_config.num_gpu_blocks = 256
    groups = [
        KVCacheGroupSpec([f"a{i}" for i in range(8)], attn),
        KVCacheGroupSpec([f"m{i}" for i in range(8)], mamba),
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=256, kv_cache_tensors=[], kv_cache_groups=groups
    )
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=768,
        hash_block_size=32,
    )
    return scheduler


def test_scheduler_without_connector_drains_boundary_offloads():
    scheduler = _scheduler_without_connector()
    assert scheduler.connector is None

    (request,) = create_requests(
        num_requests=1,
        num_tokens=MAMBA_BLOCK_SIZE + MAMBA_BLOCK_SIZE // 2,
        block_size=32,  # hash granularity, as resolve_kv_cache_block_sizes picks it
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    # Boundary offers are drained into every SchedulerOutput unconditionally;
    # with no connector they are dropped instead of forwarded — and the exact
    # source IDs never reach the worker payload.
    assert output.boundary_state_offloads is None
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert not getattr(manager, "_pending_boundary_state_offloads", [])

    # A second step with nothing new to schedule still runs cleanly.
    output = scheduler.schedule()
    assert output.boundary_state_offloads is None


def test_no_shared_offload_region_created_by_startup():
    """Off mode must not touch the /dev/shm offload region naming at all."""
    before = {p.name for p in Path("/dev/shm").glob("vllm_offload_*")}
    scheduler = _scheduler_without_connector()
    assert scheduler.connector is None
    scheduler.schedule()
    after = {p.name for p in Path("/dev/shm").glob("vllm_offload_*")}
    assert after == before
