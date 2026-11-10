# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt identity must follow a streaming session through readmission."""

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.utils import ConstantList

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("latest_only", [False, True])
@pytest.mark.parametrize("changed_token", [0, 32])
def test_same_length_session_rewrite_invalidates_prompt_identity(
    latest_only, changed_token
):
    init_none_hash(sha256)
    request = Request(
        request_id="session",
        prompt_token_ids=list(range(33)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
        resumable=True,
    )
    state = RequestOffloadState(
        config=SimpleNamespace(num_kv_cache_groups=0),
        req=request,
        req_context=SimpleNamespace(),
        offloading_context=None,
    )
    before_identity = state.prompt_identity(16)
    hashes = list(request.block_hashes)
    request.prompt_token_ids[changed_token] += 100
    request._all_token_ids[changed_token] += 100
    request.block_hashes.clear()
    request.num_computed_tokens = request.num_prompt_tokens
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler = SimpleNamespace(num_waiting_for_streaming_input=1, log_stats=False)
    update = StreamingUpdate(None, [], 8, request.arrival_time, request.sampling_params)
    Scheduler._update_request_as_session(scheduler, request, update)
    assert request.num_prompt_tokens == 33
    assert len(request.block_hashes) == len(hashes)
    if changed_token == 32:
        # Even (prompt length, hash count, final full-block hash) is unchanged.
        assert request.block_hashes == hashes
    else:
        assert request.block_hashes != hashes

    connector = object.__new__(OffloadingConnectorScheduler)
    connector.unified = SimpleNamespace(latest_only=latest_only)
    connector._req_status = {request.request_id: state}

    # Ignore object identity while comparing every existing request/state field.
    def request_fields():
        return {
            k: list(v) if isinstance(v, ConstantList) else v
            for k, v in vars(request).items()
        }

    before_request = deepcopy(request_fields())
    state_fields = [f.name for f in fields(state) if f.name != "req"]
    before_state = deepcopy({k: getattr(state, k) for k in state_fields})
    connector.update_state_after_alloc(request, None, 0)
    if latest_only:
        assert state.unified_prompt_identity is None
        assert state.prompt_identity(16) != before_identity
    else:
        assert request_fields() == before_request
        assert {k: getattr(state, k) for k in state_fields} == before_state
        assert state.prompt_identity(16) == before_identity
