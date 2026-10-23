# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sibling prefixes survive completion and bounded pressure in the host tier.

Only block metadata is simulated: allocation, sparse retention, connector
lookup/store/completion and the grouped CPU LRU are real. No tensors move.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    PrefixAnchoredSWASpec,
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK = 32
HASH_BLOCK = 8
DOC = list(range(1000, 1320))
CTX = ReqContext("pressure")

init_none_hash(sha256)


def _request(name, tokens):
    return Request(
        request_id=name,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def _store(manager, keys):
    output = manager.prepare_store(keys, CTX)
    assert output is not None
    manager.complete_store(output.keys_to_store, CTX)
    return output


def _layout(multi_group, eagle):
    full = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    groups = [KVCacheGroupSpec(["full"], full)]
    if multi_group:
        groups += [
            KVCacheGroupSpec(
                ["qsa"],
                PrefixAnchoredSWASpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                    decode_sliding_window=32,
                ),
            ),
            KVCacheGroupSpec(["draft"], full, is_eagle_group=eagle),
        ]
    else:
        groups[0].is_eagle_group = eagle
    state_groups = list(range(len(groups), len(groups) + (7 if multi_group else 1)))
    for group in state_groups:
        groups.append(
            KVCacheGroupSpec(
                [f"state{group}"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=((1,),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode="align",
                ),
            )
        )
    kv_config = KVCacheConfig(2048, [], groups)
    core = KVCacheManager(
        kv_config,
        4096,
        HASH_BLOCK,
        use_eagle=eagle,
        prefix_cache_retention_interval=0,
    )
    manager = GroupedCPUOffloadingManager(
        {
            i: CPUOffloadingManager(72 if i in state_groups else 1024)
            for i in range(len(groups))
        }
    )
    # The spec supplies configuration only; all cache operations use the real
    # grouped manager. Small shapes let this run without accelerator memory.
    spec = SimpleNamespace(
        kv_cache_config=kv_config,
        gpu_block_size=tuple(g.kv_cache_spec.block_size for g in groups),
        block_size_factor=1,
        hash_block_size=HASH_BLOCK,
        offload_prompt_only=False,
        get_manager=lambda: manager,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=1),
            cache_config=SimpleNamespace(
                prefix_cache_retention_interval=0, enable_prefix_caching=True
            ),
            speculative_config=(
                SimpleNamespace(use_eagle_kv_cache=lambda: True) if eagle else None
            ),
        ),
    )
    connector = OffloadingConnectorScheduler(spec)
    # Fill the state pools with unrelated old entries; a single new store
    # after B finishes must actually evict something in every state group.
    _store(
        manager,
        [
            make_offload_key(f"old{i}".encode(), g)
            for g in state_groups
            for i in range(72)
        ],
    )
    return core, connector, state_groups


def _complete(connector):
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: 1 for job_id in connector._jobs}
            )
        )
    )
    connector._current_batch_load_jobs.clear()


def _prefill(core, connector, req, *, known_junction=0):
    core.new_step_starts()
    computed, local = core.get_computed_blocks(req)
    assert local == 0
    connector.on_new_request(req)
    external, _ = connector.get_num_new_matched_tokens(req, local)
    assert external is not None
    if known_junction:
        # Seed an already-known document junction for the checkpoint-readback
        # path. The discovery path gets its junction from connector lookup.
        req.shared_prefix_boundary = known_junction
    state = connector._req_status[req.request_id]
    if external:
        allocated = core.allocate_slots(
            req,
            0,
            num_new_computed_tokens=local,
            new_computed_blocks=computed,
            num_external_computed_tokens=external,
            delay_cache_blocks=True,
        )
        assert allocated is not None
        # External-prefix blocks are allocated by allocate_new_computed_blocks
        # and are absent from allocate_slots' newly allocated return value.
        state.update_block_id_groups(core.get_blocks(req.request_id).get_block_ids())
        connector.update_state_after_alloc(
            req, core.get_blocks(req.request_id), external
        )
        _complete(connector)
        req.num_computed_tokens = external
        core.new_step_starts()
        core.cache_blocks(req, external)

    while req.num_computed_tokens < req.num_tokens:
        start = req.num_computed_tokens
        end = min((start // BLOCK + 1) * BLOCK, req.num_tokens)
        core.new_step_starts()
        allocated = core.allocate_slots(req, end - start)
        assert allocated is not None
        output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id=req.request_id, block_ids=allocated.get_block_ids()
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], new_block_ids=[], resumed_req_ids=set()
            ),
            num_scheduled_tokens={req.request_id: end - start},
            boundary_state_offloads=core.take_boundary_state_offloads(),
        )
        connector._build_boundary_state_store_jobs(output)
        connector._build_store_jobs(output)
        _complete(connector)
        req.num_computed_tokens = end

    req.status = RequestStatus.FINISHED_STOPPED
    connector.request_finished(req)
    core.free(req)
    # Model GPU eviction without touching the CPU tier.
    assert core.reset_prefix_cache()
    return external


@pytest.mark.parametrize("multi_group", [False, True], ids=["two-groups", "ten-groups"])
@pytest.mark.parametrize("eagle", [False, True], ids=["no-mtp", "mtp"])
@pytest.mark.parametrize("readback", [False, True], ids=["discovered", "readback"])
@pytest.mark.parametrize("suffix", [20, 192], ids=["short", "long"])
def test_sibling_cpu_hit_after_completion_and_pressure(
    multi_group, eagle, readback, suffix
):
    core, connector, state_groups = _layout(multi_group, eagle)
    boundary = len(DOC) - (BLOCK if eagle else 0)
    a = _request("a", DOC + list(range(2000, 2320)))
    assert _prefill(core, connector, a, known_junction=boundary if readback else 0) == 0

    b = _request("b", DOC + list(range(3000, 3000 + suffix)))
    hit_b = _prefill(core, connector, b)
    assert hit_b == (boundary if readback else 0)
    assert b.shared_prefix_boundary == (0 if readback else boundary)
    junction_keys = [
        make_offload_key(b.block_hashes[boundary // HASH_BLOCK - 1], g)
        for g in state_groups
    ]
    assert all(connector.manager.lookup(k, CTX) for k in junction_keys)

    pressure = _store(
        connector.manager, [make_offload_key(b"pressure", g) for g in state_groups]
    )
    assert len(pressure.evicted_keys) == len(state_groups)

    c = _request("c", DOC + [9001, 9002])
    connector.on_new_request(c)
    hit_c, _ = connector.get_num_new_matched_tokens(c, 0)
    assert hit_c == boundary
    assert all(connector.manager.lookup(k, CTX) for k in junction_keys)
