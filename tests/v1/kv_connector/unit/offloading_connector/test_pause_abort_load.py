# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from queue import Queue
from unittest.mock import Mock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_reset_lifecycle import (
    NONE,
    TP4,
)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import RequestStatus


@pytest.fixture(autouse=True)
def _gpu_platform_hma(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)


@pytest.mark.parametrize("layout", ["single", "two_group"])
@pytest.mark.parametrize("clear_cache", [False, True])
def test_pause_aborts_remote_load_before_resume(
    request_runner, layout, clear_cache, monkeypatch
):
    """Abort is immediate; GPU destinations stay owned until every rank acks."""
    h = TP4(request_runner, layout)
    h.hit(1)
    rid = h.add(h.obs * 2, 0)
    h.step(complete=NONE)
    req = h.sched.requests[rid]
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert h.cs._jobs

    core = object.__new__(EngineCoreProc)
    core.scheduler = h.sched
    core.output_queue = Queue()
    core.engines_running = False
    core.batch_queue = None
    core._idle_state_callbacks = []
    core.reset_mm_cache = Mock()
    core.reset_encoder_cache = Mock()

    future = core.pause_scheduler(mode="abort", clear_cache=clear_cache)
    assert req.status == RequestStatus.FINISHED_ABORTED
    assert h.sched.pause_state == PauseState.PAUSED_NEW
    _, outputs = core.output_queue.get_nowait()
    assert len(outputs.outputs) == 1
    assert outputs.outputs[0].request_id == rid
    assert outputs.outputs[0].finish_reason == FinishReason.ABORT
    # An output was already emitted, but freeing active DMA destinations is
    # unsafe until all TP workers report completion. No resume is needed.
    assert rid in h.sched.requests
    assert future is not None and not future.done()
    finished_by_rank: list[int] = []
    with monkeypatch.context() as observers:
        for rank, worker in enumerate(h.workers):
            original_get_finished = worker.get_finished

            def get_finished(finished_ids, rank=rank, original=original_get_finished):
                # With no live work, connector metadata requests a synchronous
                # flush. Each worker waits for its transfer, even though the
                # harness does not explicitly mark any transfer complete.
                assert rid in h.sched.requests
                assert not future.done()
                result = original(finished_ids)
                if rid in result[1]:
                    finished_by_rank.append(rank)
                return result

            observers.setattr(worker, "get_finished", get_finished)
        h.step(complete=NONE)
        assert finished_by_rank == list(range(4))
        assert rid not in h.sched.requests
    h.drain()
    h.assert_clean()
    assert h.recv_count(rid) == 1
    assert not core.has_work()
    core._notify_idle_state_callbacks()
    assert future.done()
    future.result()
    assert core.reset_mm_cache.call_count == int(clear_cache)
    assert core.reset_encoder_cache.call_count == int(clear_cache)
    assert core.is_scheduler_paused()

    # A request admitted after abort's snapshot waits until resume. HTTP
    # submission time alone cannot distinguish it from an in-flight request.
    later = h.add(h.obs * 2, 1)
    h.step()
    assert h.sched.requests[later].status == RequestStatus.WAITING
    assert not core.output_queue.qsize()
    core.resume_scheduler()
    h.miss()
    h.step()
    assert h.sched.requests[later].status == RequestStatus.RUNNING
