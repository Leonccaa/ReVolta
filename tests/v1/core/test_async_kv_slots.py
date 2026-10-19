# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loads waiting for a runner slot must not over-admit more transfers."""

import pytest

from vllm.v1.request import RequestStatus

from .test_async_kv_admission import complete, make_scheduler, request

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("second_is_load", [False, True])
def test_async_loads_occupy_concurrency_slots(second_is_load):
    scheduler = make_scheduler(blocks=1000)
    scheduler.max_num_running_reqs = 1
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (
        (64, True) if req.request_id == "a" or second_is_load else (0, False)
    )
    a = request(scheduler, "a", 80)
    b = request(scheduler, "b", 80)
    output = scheduler.schedule()
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert b.status == RequestStatus.WAITING
    assert not any(scheduler.kv_cache_manager.get_block_ids("b"))
    complete(scheduler, output, received={"a"})
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"a"}
    complete(scheduler, output)
    output = scheduler.schedule()
    assert b.status == (
        RequestStatus.WAITING_FOR_REMOTE_KVS
        if second_is_load
        else RequestStatus.RUNNING
    )


def test_cancelled_load_releases_concurrency_slot():
    scheduler = make_scheduler(blocks=1000)
    scheduler.max_num_running_reqs = 1
    a = request(scheduler, "a", 80)
    b = request(scheduler, "b", 80)
    output = scheduler.schedule()
    assert b.status == RequestStatus.WAITING
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    complete(scheduler, output, received={"a"})
    scheduler.schedule()
    assert a.is_finished() and b.status == RequestStatus.WAITING_FOR_REMOTE_KVS
