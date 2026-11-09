# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A broken external KV load is recomputed, never crashes the engine.

With ``load_failure_recompute_enabled`` (the cross-group hit contract or the
long-context mode), three failures void a request's external hit instead of
raising or asserting:

* a key is evicted after the hit was committed (between lookup and the load
  plan),
* only some KV cache groups are still readable when the load is planned,
* a worker reports the load transfer as failed.

The scheduler then releases the request's blocks (none of the loaded blocks
was published to the prefix cache), schedules it again from its local hit
with the external store skipped once, and it finishes normally. Other
requests are not touched.
"""

from typing import Any

import pytest
import torch

import tests.v1.kv_connector.unit.offloading_connector.utils as runner_utils
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.config.kernel import LongContextKVConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import get_offload_group_idx
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.request import RequestStatus

BLOCK_SIZE = 4
NUM_GPU_BLOCKS = 100


def _groups() -> list[KVCacheGroupSpec]:
    return [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=2 * BLOCK_SIZE,
            ),
        ),
    ]


def _runner(request_runner, monkeypatch, gate: str, async_scheduling: bool):
    """A two-group runner with the recompute path switched on by ``gate``."""
    create = runner_utils.create_vllm_config

    def create_with_gate(*args: Any, **kwargs: Any):
        config = create(*args, **kwargs)
        if gate == "contract":
            config.kernel_config.qsa_cross_group_hit_check = True
        elif gate == "long_context":
            config.kernel_config.long_context_kv = LongContextKVConfig(
                enabled=True, host_pool_gib=1.0
            )
        return config

    monkeypatch.setattr(runner_utils, "create_vllm_config", create_with_gate)
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        kv_cache_groups=_groups(),
    )
    assert runner.scheduler.recompute_failed_loads is (gate != "off")
    # The tier holds exactly what was stored (runner.tier).
    runner.tier = set()

    def store(keys, req_context):
        keys = list(keys)
        runner.tier.update(keys)
        return generate_store_output(keys)

    runner.manager.prepare_store.side_effect = store
    runner.manager.lookup.side_effect = lambda key, req_context: key in runner.tier
    return runner


class _Recorder:
    """Watches a runner's scheduler for voided loads and released blocks."""

    def __init__(self, runner):
        self.runner = runner
        scheduler = runner.scheduler
        self.voided: list[str] = []
        self.unpublished_release: list[bool] = []
        void = scheduler._void_failed_loads

        def record_void(req_ids):
            req_ids = set(req_ids)
            self.voided.extend(sorted(req_ids))
            return void(req_ids)

        scheduler._void_failed_loads = record_void
        manager = scheduler.kv_cache_manager
        free = manager.free

        def record_free(request):
            if request.request_id in scheduler.failed_recving_kv_req_ids:
                # The failed load's blocks must leave without a prefix hash:
                # nothing restored by it may serve another request.
                self.unpublished_release.append(
                    all(
                        block.block_hash is None
                        for blocks in manager.coordinator.get_blocks(request.request_id)
                        for block in blocks
                    )
                )
            return free(request)

        manager.free = record_free

        handler = runner.offloading_spec.handler
        self.load_jobs: list[int] = []
        transfer = handler.transfer_async

        def record_transfer(job_id, spec):
            src, _ = spec
            if isinstance(src, runner_utils.MockLoadStoreSpec):
                self.load_jobs.append(job_id)
            return transfer(job_id, spec)

        handler.transfer_async = record_transfer


def _store_prefix(runner, num_blocks: int) -> list[int]:
    """Run one request to completion so its prompt is offloaded, then drop
    the GPU prefix cache so the next identical prompt hits only the tier."""
    prompt = [0] * (BLOCK_SIZE * num_blocks)
    runner.new_request(token_ids=prompt)
    runner._run(decoded_tokens=[0, 0, EOS_TOKEN_ID], complete_transfers=True)
    runner.completed_loads.clear()
    runner.completed_stores.clear()
    runner.flushed_gpu_blocks.clear()
    assert not runner.scheduler.requests
    # Both groups of every prompt block reached the tier.
    assert {get_offload_group_idx(k) for k in runner.tier} == {0, 1}
    assert runner.scheduler.reset_prefix_cache()
    return prompt + [0]


def _check_pool_drained(runner) -> None:
    for pool in (runner.scheduler.kv_cache_manager.coordinator.block_pool,):
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def _finish(runner, recorder: _Recorder) -> None:
    runner._run(decoded_tokens=[0, 0, EOS_TOKEN_ID], complete_transfers=True)
    runner.completed_loads.clear()
    runner.completed_stores.clear()
    runner.flushed_gpu_blocks.clear()
    assert not runner.scheduler.requests
    _check_pool_drained(runner)


@pytest.mark.parametrize("gate", ["contract", "long_context"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_key_evicted_after_commit_recomputes(
    request_runner, monkeypatch, gate: str, async_scheduling: bool
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling)
    recorder = _Recorder(runner)
    prompt = _store_prefix(runner, 3)

    cs = runner.connector_scheduler
    plan = cs.update_state_after_alloc

    def evict_then_plan(request, blocks, num_external_tokens):
        if num_external_tokens:
            runner.tier.clear()  # the tier lost every key after the lookup
        return plan(request, blocks, num_external_tokens)

    cs.update_state_after_alloc = evict_then_plan
    runner.new_request(token_ids=prompt)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert recorder.load_jobs == []  # nothing was read back
    # Abandoned at plan time: parked as a finished failed receive directly.
    assert cs.num_abandoned_loads == 1
    assert recorder.voided == []
    assert recorder.unpublished_release == [True]
    assert not cs._skip_external_once  # the retry consumed it


@pytest.mark.parametrize("gate", ["contract", "long_context"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_one_group_missing_at_plan_time_recomputes(
    request_runner, monkeypatch, gate: str, async_scheduling: bool
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling)
    recorder = _Recorder(runner)
    prompt = _store_prefix(runner, 3)

    cs = runner.connector_scheduler
    plan = cs.update_state_after_alloc

    def lose_window_group(request, blocks, num_external_tokens):
        if num_external_tokens:
            # Full attention is still there; the sliding-window group is not.
            runner.tier = {k for k in runner.tier if get_offload_group_idx(k) == 0}
        return plan(request, blocks, num_external_tokens)

    cs.update_state_after_alloc = lose_window_group
    runner.new_request(token_ids=prompt)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert recorder.load_jobs == []
    assert cs.num_abandoned_loads == 1
    assert recorder.voided == []
    assert recorder.unpublished_release == [True]


@pytest.mark.parametrize("gate", ["contract", "long_context"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_failed_load_transfer_recomputes_only_that_request(
    request_runner, monkeypatch, gate: str, async_scheduling: bool
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling)
    recorder = _Recorder(runner)
    prompt = _store_prefix(runner, 3)

    handler = runner.offloading_spec.handler
    complete = handler.complete_jobs

    def fail_loads(job_ids):
        complete(job_ids)
        handler.completed_transfers = [
            TransferResult(
                job_id=r.job_id,
                success=r.job_id not in recorder.load_jobs,
                transfer_size=None,
                transfer_time=None,
                transfer_type=None,
            )
            for r in handler.completed_transfers
        ]

    handler.complete_jobs = fail_loads
    runner.new_request(token_ids=prompt)
    failed = runner.scheduler.requests[str(runner.req_id)]
    # A request with another prompt runs in the same steps and misses.
    runner.new_request(token_ids=[7] * (BLOCK_SIZE * 2 + 1))
    other = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)

    assert len(recorder.load_jobs) == 1  # one load, and it failed
    assert recorder.voided == [failed.request_id]
    assert recorder.unpublished_release == [True]
    assert failed.status == other.status == RequestStatus.FINISHED_STOPPED
    assert other.num_preemptions == 0
    cs = runner.connector_scheduler
    assert cs.num_abandoned_loads == 1
    # The failed job's bookkeeping was released like a completed one.
    assert not cs._jobs


def test_disabled_gate_keeps_the_assert(request_runner, monkeypatch):
    runner = _runner(request_runner, monkeypatch, "off", async_scheduling=False)
    prompt = _store_prefix(runner, 2)
    handler = runner.offloading_spec.handler
    complete = handler.complete_jobs

    def fail_all(job_ids):
        complete(job_ids)
        handler.completed_transfers = [
            TransferResult(
                job_id=r.job_id,
                success=False,
                transfer_size=None,
                transfer_time=None,
                transfer_type=None,
            )
            for r in handler.completed_transfers
        ]

    handler.complete_jobs = fail_all
    runner.new_request(token_ids=prompt)
    with pytest.raises(AssertionError):
        runner._run(decoded_tokens=[0, EOS_TOKEN_ID], complete_transfers=True)
