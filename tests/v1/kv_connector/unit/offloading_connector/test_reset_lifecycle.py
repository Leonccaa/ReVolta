# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID, create_model_runner_output
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

NUM_WORKERS = 4


BS = 4


ALL = (True,) * NUM_WORKERS


NONE = (False,) * NUM_WORKERS


@pytest.fixture(autouse=True)
def _gpu_platform_hma(monkeypatch):
    # The CPU test container resolves a non-GPU platform, which refuses the
    # hybrid KV cache manager that RequestRunner enables explicitly (the
    # upstream offloading tests fail here for that reason alone).
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)


def _two_groups():
    return [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=BS,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=8,
            ),
        ),
    ]


class TP4:
    """Scheduler and four connector workers with mock transfer handlers."""

    def __init__(self, request_runner, layout: str):
        self.layout = layout
        groups = None if layout == "single" else _two_groups()
        factor = 3 if layout == "single" else 1
        self.obs = BS * factor
        self.r = request_runner(
            block_size=BS,
            num_gpu_blocks=100,
            async_scheduling=False,
            block_size_factor=factor,
            kv_cache_groups=groups,
        )
        self.sched = self.r.scheduler
        self.cs = self.r.connector_scheduler
        self.cs.config = self.cs.config._replace(num_workers=NUM_WORKERS)
        vcfg = self.sched.vllm_config
        kcfg = self.sched.kv_cache_config
        self.workers = [self.r.worker_connector]
        for _ in range(NUM_WORKERS - 1):
            w = OffloadingConnector(vcfg, KVConnectorRole.WORKER, kcfg)
            kv = {}
            for g in kcfg.kv_cache_groups:
                s = g.kv_cache_spec
                for ln in g.layer_names:
                    kv[ln] = torch.empty(
                        kcfg.num_blocks,
                        2,
                        s.block_size,
                        s.num_kv_heads,
                        s.head_size,
                        dtype=s.dtype,
                    )
            with set_current_vllm_config(vcfg):
                w.register_kv_caches(kv)
            self.workers.append(w)
        self.agg = KVOutputAggregator(NUM_WORKERS)
        self.recv_log: list[tuple[int, str]] = []
        self.nstep = 0
        self.r.manager.prepare_store.side_effect = lambda keys, ctx: (
            generate_store_output(keys)
        )

    # CPU tier hit for the next lookups
    def hit(self, n_offloaded_blocks: int):
        if self.layout == "single":
            self.cs._maximal_prefix_lookup = lambda key, req_context: n_offloaded_blocks
        else:
            self.r.manager.lookup.return_value = True

    def miss(self):
        if self.layout == "single":
            self.cs._maximal_prefix_lookup = lambda key, req_context: 0
        else:
            self.r.manager.lookup.return_value = False

    def add(self, n_tokens: int, tok: int) -> str:
        self.r.new_request(token_ids=[tok] * n_tokens)
        return str(self.r.req_id)

    def step(self, complete=ALL, token: int = 0, extra=None):
        so = self.sched.schedule()
        meta = so.kv_connector_metadata
        outs = []
        for i, w in enumerate(self.workers):
            w.handle_preemptions(meta)
            w.bind_connector_metadata(meta)
            w.start_load_kv(self.r._dummy_ctx)
            if complete[i]:
                w.connector_worker.spec.complete_transfers()
            fs, fr = w.get_finished(so.finished_req_ids)
            wm = w.build_connector_worker_meta() or OffloadingWorkerMetadata()
            w.clear_connector_metadata()
            outs.append(
                create_model_runner_output(
                    reqs=self.sched.running,
                    finished_sending=fs,
                    finished_recving=fr,
                    token_id=token,
                    kv_connector_worker_meta=wm,
                )
            )
        out = self.agg.aggregate(outs, 0)
        kvo = out.kv_connector_output
        if extra is not None:
            extra(kvo)
        for rid in (kvo.finished_recving or ()) if kvo else ():
            self.recv_log.append((self.nstep, rid))
        self.sched.update_from_output(so, out)
        self.nstep += 1
        return so

    def recv_count(self, rid: str) -> int:
        return sum(1 for _, r in self.recv_log if r == rid)

    def drain(self, n: int = 6):
        for _ in range(n):
            self.step()

    def assert_clean(self):
        assert not self.sched.requests, self.sched.requests
        assert not self.cs._jobs, self.cs._jobs
        assert not self.cs._req_status, self.cs._req_status


LAYOUTS = ["single", "two_group"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_preempt_with_pending_stores_then_reload(request_runner, layout):
    h = TP4(request_runner, layout)
    fbq = h.sched.kv_cache_manager.block_pool.free_block_queue
    empty = fbq.num_free_blocks
    h.miss()
    rid = h.add(h.obs * 2, 0)
    skew = (True, False, False, False)
    h.step(complete=skew)  # prefill, stores issued next step
    for _ in range(2 * h.obs - BS):
        h.step(complete=skew, token=0)
    req = h.sched.requests[rid]
    assert h.cs._req_status[rid].transfer_jobs, "expected pending stores"
    fbq.num_free_blocks = 0
    for _ in range(4):
        h.step(complete=skew, token=0)
        if req.status == RequestStatus.PREEMPTED:
            break
    assert req.status == RequestStatus.PREEMPTED
    assert not h.cs._req_status[rid].transfer_jobs  # flushed on preemption
    fbq.num_free_blocks = empty
    assert h.sched.reset_prefix_cache()  # GPU prefix cache only (upstream test)
    h.hit(2)
    h.step(complete=(True, False, False, False))  # reload issued
    h.step(complete=(False, True, True, True))
    assert h.recv_count(rid) == 1
    for _ in range(3):
        h.step(token=0)
    h.step(token=EOS_TOKEN_ID)
    h.drain()
    assert h.recv_count(rid) == 1
    h.assert_clean()


def test_dev_reset_cache_with_inflight_load(request_runner):
    """Reset connector state while a CPU load is in flight."""
    h = TP4(request_runner, "single")
    h.hit(1)
    rid = h.add(h.obs * 2, 0)
    h.step(complete=NONE)  # load in flight
    h.sched.reset_prefix_cache(reset_connector=True)
    h.miss()
    h.step()  # flushed load completes, request promoted next step
    assert h.recv_count(rid) == 1
    for _ in range(2 * h.obs):
        h.step(token=0)  # stores for the request
    h.step(token=EOS_TOKEN_ID)
    h.drain()
    assert h.recv_count(rid) == 1
    h.assert_clean()


@pytest.mark.parametrize("reset_connector", [False, True])
def test_dev_reset_running_requests_with_pending_stores(
    request_runner, reset_connector
):
    """reset_prefix_cache(reset_running_requests=True) preempts running
    requests outside schedule(), so preempted_req_ids never reaches the
    connector and their pending stores are not flushed (dev mode only)."""
    h = TP4(request_runner, "single")
    h.miss()
    rid = h.add(h.obs * 2, 0)
    skew = (True, False, False, False)
    h.step(complete=skew)
    for _ in range(2 * h.obs - BS):
        h.step(complete=skew, token=0)
    assert h.cs._req_status[rid].transfer_jobs, "expected pending stores"
    pending = set(h.cs._req_status[rid].transfer_jobs)
    assert h.sched.reset_prefix_cache(
        reset_running_requests=True, reset_connector=reset_connector
    )
    if reset_connector:
        h.miss()
    else:
        h.hit(2)
    so = h.step(complete=skew)
    assert rid in so.preempted_req_ids
    assert pending <= so.kv_connector_metadata.jobs_to_flush
    if not reset_connector:
        assert not so.kv_connector_metadata.load_jobs
    # A connector reset restarts store progress, so a resumed request may
    # already have fresh stores. Only the old transfer IDs must be gone.
    assert pending.isdisjoint(h.cs._req_status[rid].transfer_jobs)
    for _ in range(3):
        h.step(token=0)
    h.step(token=EOS_TOKEN_ID)
    h.drain()
    h.assert_clean()


def test_reset_running_flushes_pending_stores(request_runner):
    """Even with #46231's deferral, forced preemption must reach workers."""
    h = TP4(request_runner, "single")
    h.miss()
    rid = h.add(h.obs * 2, 0)
    skew = (True, False, False, False)
    h.step(complete=skew)
    for _ in range(2 * h.obs - BS):
        h.step(complete=skew, token=0)
    pending = set(h.cs._req_status[rid].transfer_jobs)
    assert pending
    assert h.sched.reset_prefix_cache(reset_running_requests=True)
    h.hit(2)
    so = h.sched.schedule()
    assert rid in so.preempted_req_ids, "forced preemption absent from scheduler output"
    assert pending <= so.kv_connector_metadata.jobs_to_flush


def test_reset_finalizes_finished_status(request_runner):
    h = TP4(request_runner, "single")
    h.miss()
    h.add(h.obs, 7)  # another live request prevents automatic flush-all
    h.step()
    rid = h.add(h.obs * 2, 0)
    skew = (True, False, False, False)
    h.step(complete=skew)
    for _ in range(2 * h.obs - BS):
        h.step(complete=skew, token=0)
    assert h.cs._req_status[rid].transfer_jobs
    h.sched.finish_requests((rid,), RequestStatus.FINISHED_ABORTED)
    assert h.cs._req_status[rid].req.is_finished()
    h.sched.reset_prefix_cache(reset_connector=True)
    assert rid not in h.cs._req_status, "finished request state survives reset"


@pytest.mark.parametrize("reset_connector", [False, True])
def test_reset_running_during_remote_load_keeps_busy_contract(
    request_runner, reset_connector
):
    h = TP4(request_runner, "single")
    h.hit(1)
    rid = h.add(h.obs * 2, 0)
    h.step(complete=NONE)
    # A live remote load owns GPU destinations. Preserve the existing busy
    # rejection rather than resetting the pool underneath that transfer.
    with pytest.raises(RuntimeError, match="waiting for remote KV transfer"):
        h.sched.reset_prefix_cache(
            reset_running_requests=True, reset_connector=reset_connector
        )
    h.miss()
    h.step()
    h.step(token=EOS_TOKEN_ID)
    h.drain()
    assert h.recv_count(rid) == 1
    h.assert_clean()
