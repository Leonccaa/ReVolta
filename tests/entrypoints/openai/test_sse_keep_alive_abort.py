# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise AsyncLLM's real cancellation/abort chain with CPU transport doubles."""

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import StreamingResponse

from vllm.entrypoints.openai.sse_keep_alive import with_sse_keep_alive
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

pytestmark = [pytest.mark.asyncio, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("interval", [0, 0.01])
@pytest.mark.parametrize("transport_checkpoint", [False, True])
async def test_disconnect_runs_real_generate_abort_and_closes_queue(
    interval, transport_checkpoint
):
    waiting = asyncio.Event()
    state = SimpleNamespace(core_sent=False)

    async def get():
        waiting.set()
        await asyncio.Event().wait()

    async def abort_requests_async(request_ids):
        assert request_ids == ["internal"]
        if transport_checkpoint:
            await asyncio.sleep(0)
        state.core_sent = True

    queue = SimpleNamespace(
        request_id="internal", get_nowait=lambda: None, get=get, close=MagicMock()
    )
    engine = SimpleNamespace(
        add_request=AsyncMock(return_value=queue),
        output_processor=SimpleNamespace(
            abort_requests=MagicMock(return_value=["internal"])
        ),
        engine_core=SimpleNamespace(
            abort_requests_async=AsyncMock(side_effect=abort_requests_async)
        ),
        _mm_shadow_hits={"internal": "cached"},
        log_requests=False,
    )
    engine.abort = partial(AsyncLLM.abort, engine)
    generator = AsyncLLM.generate(
        engine, "test prompt", SamplingParams(max_tokens=1), "external"
    )
    response = StreamingResponse(
        with_sse_keep_alive(generator, interval), media_type="text/event-stream"
    )

    async def receive():
        await waiting.wait()
        await asyncio.sleep(0.02)
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    await asyncio.wait_for(
        response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send),
        2,
    )
    engine.output_processor.abort_requests.assert_called_once_with(("internal",), True)
    engine.engine_core.abort_requests_async.assert_awaited_once_with(["internal"])
    queue.close.assert_called_once()
    assert not engine._mm_shadow_hits
    # Starlette's level cancellation can interrupt an awaited core send. This
    # existing limitation applies with and without keep-alive; a synchronous
    # transport submission completes in both cases.
    assert state.core_sent is (not transport_checkpoint)
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []
